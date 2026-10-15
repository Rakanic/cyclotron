//! Per-warp pipeline hazards of the Muon core, as the RTL exposes them: what keeps a warp's next instruction from
//! issuing besides the shared functional units and memory queues the rest of the timing model already has.
//!
//! * Register dependences (RAW): an instruction issues only once its source registers are written; a result is
//!   ready `latency(class)` cycles after its producer issued.
//! * Control-flow redirects: after a branch, jump, SIMT split / pred, or CSR access, the warp's next
//!   instruction issues `redirect` cycles later (the front end refetches). `vx_join` has its own `join`: the RTL
//!   never issues it (it is absent from the RTL trace) and it adds ~3 cycles to the redirect before it.
//! * The divider is not pipelined: a divide holds its warp for its full latency.
//! * `fence.s` (MISC_MEM f3 = 2) only orders the warp's shared-memory accesses -- a "fake" store-queue entry in the
//!   RTL LSU (LSU.scala:570-572) -- so it costs `fence_smem` cycles and sends no cache flush.
//!
//! Values: measured on the RTL trace of fa_mxfp8_sr (VCS RadianceE4M3MxGemminiSingleClusterTraceConfig,
//! 2026-10-07; npu-exploration tools/radiance_trace_compare.py latency): cycles from an instruction to the next one
//! of its warp, when that one reads its result (dependent) or not (independent); see
//! config/timing/radiance/hazards.toml.
//!
//! Off unless the timing config has `[hazards] enabled = true`.

use crate::muon::decode::DecodedInst;
use crate::muon::execute::Opcode;
use crate::timeq::Cycle;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct HazardConfig {
    pub enabled: bool,
    /// result latencies (issue -> a dependent instruction may issue)
    pub alu: Cycle,
    pub mul: Cycle,
    pub fp: Cycle,
    pub fp_half: Cycle,
    pub sfu: Cycle,
    pub load_shared: Cycle,
    /// a global load's minimum (an L0 hit); misses are the gmem model's
    pub load_global: Cycle,
    /// a divide holds its warp this long (and its result is ready then)
    pub div: Cycle,
    /// branch / jump / SIMT control / CSR -> the warp's next instruction
    pub redirect: Cycle,
    /// vx_join -> the warp's next instruction (the RTL front end pops the reconvergence stack without issuing it)
    pub join: Cycle,
    /// fence.s -> the warp's next instruction
    pub fence_smem: Cycle,
    /// a warp's global stores retire in order, each on its memory response; its next global load or store waits
    pub store_in_order: bool,
}

impl Default for HazardConfig {
    /// The calibrated values (config/timing/radiance/hazards.toml): the RTL measurement minus the one cycle a
    /// stalled warp's replay costs cyclotron. They assume execute units that do not block their warp
    /// (config/timing/radiance/execute.toml).
    fn default() -> Self {
        Self {
            enabled: false,
            alu: 3,          // RTL dependent 4 (independent 2)
            mul: 36,         // 37 (few samples)
            fp: 6,           // FP32 dependent 7
            fp_half: 5,      // BF16 dependent 6
            sfu: 12,         // fexp.h dependent 13
            load_shared: 12, // lw.shared -> use 13
            load_global: 15, // lw.global -> use 16
            div: 37,         // 38, dependent or not
            redirect: 8,     // branch / jump / vx_split / CSR -> next 9
            join: 8,         // as a redirect unless configured (RTL: +3 on the preceding redirect)
            fence_smem: 9,   // fence.s -> next 10
            store_in_order: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Class {
    Alu,
    Mul,
    Div,
    Fp,
    FpHalf,
    Sfu,
    LoadShared,
    LoadGlobal,
    Redirect,
    Join,
    FenceSmem,
    Other, // stores, barriers, plain fences: no result, no redirect
}

fn classify(inst: &DecodedInst) -> Class {
    match inst.opcode {
        Opcode::LOAD | Opcode::LOAD_FP => {
            if inst.opext == 1 {
                Class::LoadShared
            } else {
                Class::LoadGlobal
            }
        }
        Opcode::OP_IMM | Opcode::AUIPC | Opcode::LUI => Class::Alu,
        Opcode::OP if inst.f7 == 1 => {
            if inst.f3 >= 4 {
                Class::Div
            } else {
                Class::Mul
            }
        }
        Opcode::OP => Class::Alu,
        Opcode::OP_FP | Opcode::MADD | Opcode::MSUB | Opcode::NM_SUB | Opcode::NM_ADD => {
            if inst.f7 & 0b11 == 0b10 {
                Class::FpHalf
            } else {
                Class::Fp
            }
        }
        Opcode::CUSTOM3 => Class::Sfu,
        Opcode::CUSTOM0 if inst.f3 == 3 => Class::Join, // vx_join
        Opcode::BRANCH | Opcode::JAL | Opcode::JALR | Opcode::CUSTOM0 | Opcode::SYSTEM => Class::Redirect,
        Opcode::MISC_MEM if inst.f3 == 2 => Class::FenceSmem,
        _ => Class::Other,
    }
}

/// The registers the instruction reads (0 = none / x0)
fn sources(inst: &DecodedInst) -> [u8; 3] {
    let r = inst.has_regs();
    let mut s = [
        if r.rs1 { inst.rs1_addr } else { 0 },
        if r.rs2 { inst.rs2_addr } else { 0 },
        if r.rs3 { inst.rs3_addr } else { 0 },
    ];
    match inst.opcode {
        // Muon's custom formats are not in has_regs(): SIMT control (vx_split_n a0, a2 reads a2), fexp, CSR, barriers
        Opcode::CUSTOM0 | Opcode::CUSTOM2 | Opcode::CUSTOM3 => {
            s[0] = inst.rs1_addr;
            s[1] = inst.rs2_addr;
        }
        Opcode::SYSTEM if (1..=3).contains(&inst.f3) => s[0] = inst.rs1_addr,
        _ => {}
    }
    s
}

fn writes_rd(inst: &DecodedInst) -> bool {
    inst.rd_addr != 0
        && !matches!(
            inst.opcode,
            Opcode::STORE | Opcode::STORE_FP | Opcode::BRANCH | Opcode::MISC_MEM
        )
}

/// One core's warps
pub struct WarpHazards {
    config: HazardConfig,
    reg_ready: Vec<Vec<Cycle>>, // [warp][register]: the cycle a dependent instruction may issue
    next_issue: Vec<Cycle>,     // [warp]: redirects, a running divide, fence.s
    stores_out: Vec<u32>,       // [warp]: global stores sent and not yet answered
    pub stalls: u64,            // instruction issues it delayed
}

impl WarpHazards {
    pub fn new(config: HazardConfig, num_warps: usize) -> Self {
        Self {
            config,
            reg_ready: vec![vec![0; 256]; num_warps],
            next_issue: vec![0; num_warps],
            stores_out: vec![0; num_warps],
            stalls: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    /// fence.s is handled here (ordering only): the core must not send its cache flush
    pub fn is_smem_fence(&self, inst: &DecodedInst) -> bool {
        self.config.enabled && classify(inst) == Class::FenceSmem
    }

    /// The first cycle `inst` may issue for warp `wid` (<= now: it may issue now)
    pub fn ready_at(&self, wid: usize, inst: &DecodedInst) -> Cycle {
        if !self.config.enabled {
            return 0;
        }
        sources(inst)
            .iter()
            .filter(|&&r| r != 0)
            .map(|&r| self.reg_ready[wid][r as usize])
            .fold(self.next_issue[wid], Cycle::max)
    }

    /// A global memory request of warp `wid` must wait for the warp's stores to retire
    pub fn gmem_blocked(&self, wid: usize) -> bool {
        self.config.enabled && self.config.store_in_order && self.stores_out[wid] > 0
    }

    /// A global store of warp `wid` was sent (`done` = false) or answered (`done` = true)
    pub fn store_out(&mut self, wid: usize, done: bool) {
        if self.config.enabled && self.config.store_in_order {
            let n = &mut self.stores_out[wid];
            *n = if done { n.saturating_sub(1) } else { *n + 1 };
        }
    }

    /// A delayed issue (for the stall count)
    pub fn note_stall(&mut self) {
        self.stalls += 1;
    }

    /// `inst` issued for warp `wid` at `now`
    pub fn issued(&mut self, wid: usize, inst: &DecodedInst, now: Cycle) {
        if !self.config.enabled {
            return;
        }
        let c = &self.config;
        let cls = classify(inst);
        let latency = match cls {
            Class::Alu => c.alu,
            Class::Mul => c.mul,
            Class::Div => c.div,
            Class::Fp => c.fp,
            Class::FpHalf => c.fp_half,
            Class::Sfu => c.sfu,
            Class::LoadShared => c.load_shared,
            Class::LoadGlobal => c.load_global,
            Class::Redirect | Class::Join => c.alu,
            Class::FenceSmem | Class::Other => 0,
        };
        if writes_rd(inst) && latency > 0 {
            self.reg_ready[wid][inst.rd_addr as usize] = now + latency;
        }
        let hold = match cls {
            Class::Redirect => c.redirect,
            Class::Join => c.join,
            Class::Div => c.div,
            Class::FenceSmem => c.fence_smem,
            _ => 0,
        };
        if hold > 0 {
            self.next_issue[wid] = self.next_issue[wid].max(now + hold);
        }
    }
}
