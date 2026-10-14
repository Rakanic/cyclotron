//! MX-Gemmini in the cluster: libgemmini's functional + timing model (libgemmini/gemmini_radiance.h) behind the
//! cluster's Gemmini MMIO registers (GPU-local 0x84000, radiance GemminiTile.scala:360-390).
//!
//! Muon reaches the registers with `sw.shared` / `lw.shared`. A store to INST is not taken while Gemmini's command
//! path is full (the RTL's io.cmd.ready): with the timing model the warp replays it, as the RTL store waits. Gemmini's
//! DMA reads and writes global memory, and its scratchpad is this cluster's shared memory (row r at byte 16 r).
//!
//! Configured by the `[gemmini]` section of config.toml; needs `cargo build --features gemmini`.

use crate::sim::config::Config;
use crate::sim::flat_mem::FlatMemory;
use crate::timeq::Cycle;
use serde::Deserialize;
use std::sync::{Arc, RwLock};

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct GemminiConfig {
    /// Model the cluster's Gemmini (otherwise its registers are plain shared memory)
    pub enabled: bool,
    /// "both": bit-exact functional + timing; "perf": timing only (data-moving commands skip their work)
    pub mode: String,
    /// Timing preset (libgemmini perf/params/config.cc) and "section.name=value,..." overrides
    pub preset: String,
    pub overrides: String,
    /// The register block (cluster-local shared-memory address) and its size
    pub mmio_base: u32,
    pub mmio_size: u32,
    /// The timing report at the end of the run (empty: the perf log directory, else stderr)
    pub report: String,
}

impl Default for GemminiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "both".into(),
            preset: "radiance".into(),
            overrides: String::new(),
            mmio_base: 0x84000,
            mmio_size: 0x1000,
            report: String::new(),
        }
    }
}

impl Config for GemminiConfig {}

#[cfg_attr(not(feature = "gemmini"), allow(dead_code))]
const R_INST: u32 = 0x00;

#[cfg(feature = "gemmini")]
mod ffi {
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct Soc {
        pub ctx: *mut c_void,
        pub load: extern "C" fn(*mut c_void, u64, u32) -> u64,
        pub store: extern "C" fn(*mut c_void, u64, u64, u32),
        pub smem: *mut u8,
        pub smem_bytes: u64,
    }

    #[repr(C)]
    pub struct Device {
        _private: [u8; 0],
    }

    extern "C" {
        pub fn gemmini_radiance_new(
            mode: c_int,
            preset: *const c_char,
            overrides: *const c_char,
            soc: *const Soc,
        ) -> *mut Device;
        pub fn gemmini_radiance_free(g: *mut Device);
        pub fn gemmini_radiance_write(g: *mut Device, cycle: u64, offset: u32, value: u64, bytes: u32) -> c_int;
        pub fn gemmini_radiance_read(g: *mut Device, cycle: u64, offset: u32, bytes: u32) -> u64;
        pub fn gemmini_radiance_busy(g: *mut Device, cycle: u64) -> c_int;
        pub fn gemmini_radiance_finish(g: *mut Device, report_path: *const c_char) -> u64;
    }
}

/// Gemmini's view of global memory: the address its DMA uses (0x1_0000_0000 | GPU address)
#[cfg_attr(not(feature = "gemmini"), allow(dead_code))]
struct GmemPort {
    gmem: Arc<RwLock<FlatMemory>>,
}

#[cfg(feature = "gemmini")]
extern "C" fn gmem_load(ctx: *mut std::os::raw::c_void, addr: u64, bytes: u32) -> u64 {
    use crate::base::mem::HasMemory;
    let port = unsafe { &*(ctx as *const GmemPort) };
    let gmem = port.gmem.read().expect("gmem poisoned");
    let data = gmem
        .read_impl((addr & 0xFFFF_FFFF) as usize, bytes as usize)
        .expect("gemmini DMA load out of range");
    data.iter().rev().fold(0u64, |v, b| (v << 8) | *b as u64)
}

#[cfg(feature = "gemmini")]
extern "C" fn gmem_store(ctx: *mut std::os::raw::c_void, addr: u64, value: u64, bytes: u32) {
    use crate::base::mem::HasMemory;
    let port = unsafe { &*(ctx as *const GmemPort) };
    let mut gmem = port.gmem.write().expect("gmem poisoned");
    gmem.write_impl((addr & 0xFFFF_FFFF) as usize, &value.to_le_bytes()[..bytes as usize])
        .expect("gemmini DMA store out of range");
}

/// One cluster's Gemmini
pub struct GemminiDevice {
    config: GemminiConfig,
    #[cfg(feature = "gemmini")]
    dev: *mut ffi::Device,
    _port: Box<GmemPort>,
    now: Cycle,
    pub commands: u64,
    pub refused: u64,
}

// The device is only ever used under its cluster's lock, from the simulation thread.
unsafe impl Send for GemminiDevice {}

impl GemminiDevice {
    /// The cluster's cycle; register accesses use it (it never goes back).
    pub fn set_now(&mut self, now: Cycle) {
        self.now = self.now.max(now);
    }

    pub fn matches(&self, addr: u32) -> bool {
        addr >= self.config.mmio_base && addr < self.config.mmio_base + self.config.mmio_size
    }

    fn report_path(&self, default_report: Option<std::path::PathBuf>) -> Option<std::path::PathBuf> {
        if !self.config.report.is_empty() {
            Some(std::path::PathBuf::from(&self.config.report))
        } else {
            default_report
        }
    }
}

#[cfg(feature = "gemmini")]
impl GemminiDevice {
    /// `smem`: the cluster's shared memory; it must outlive the device and never move (FlatMemory never resizes).
    pub fn new(config: &GemminiConfig, gmem: Arc<RwLock<FlatMemory>>, smem: &mut FlatMemory) -> Self {
        use std::ffi::CString;
        let port = Box::new(GmemPort { gmem });
        let soc = ffi::Soc {
            ctx: &*port as *const GmemPort as *mut _,
            load: gmem_load,
            store: gmem_store,
            smem: smem.as_mut_ptr(),
            smem_bytes: smem.len() as u64,
        };
        let mode = match config.mode.as_str() {
            "both" => 1,
            "perf" => 0,
            m => panic!("[gemmini] mode = \"{}\": want \"both\" or \"perf\"", m),
        };
        let preset = CString::new(config.preset.clone()).unwrap();
        let overrides = CString::new(config.overrides.clone()).unwrap();
        let dev = unsafe { ffi::gemmini_radiance_new(mode, preset.as_ptr(), overrides.as_ptr(), &soc) };
        assert!(!dev.is_null(), "gemmini_radiance_new failed (shared memory smaller than the scratchpad?)");
        GemminiDevice { config: config.clone(), dev, _port: port, now: 0, commands: 0, refused: 0 }
    }

    /// Would a store to `addr` be taken at `now`? (false only for INST while the command path is full)
    pub fn store_ready(&mut self, now: Cycle, addr: u32) -> bool {
        self.set_now(now);
        addr - self.config.mmio_base != R_INST || unsafe { ffi::gemmini_radiance_read(self.dev, self.now, 0x08, 4) != 0 }
    }

    /// A store to a register. A refused INST store waits here, the cycle moving on (without the timing model, where
    /// the warp cannot replay it; with it, store_ready() said yes this cycle).
    pub fn store(&mut self, addr: u32, value: u32, bytes: u32) {
        let off = addr - self.config.mmio_base;
        while unsafe { ffi::gemmini_radiance_write(self.dev, self.now, off, value as u64, bytes) } == 0 {
            self.refused += 1;
            self.now += 1;
        }
        if off == R_INST {
            self.commands += 1;
        }
    }

    pub fn load(&mut self, addr: u32, bytes: u32) -> u32 {
        let off = addr - self.config.mmio_base;
        unsafe { ffi::gemmini_radiance_read(self.dev, self.now, off, bytes) as u32 }
    }

    pub fn busy(&mut self, now: Cycle) -> bool {
        self.set_now(now);
        unsafe { ffi::gemmini_radiance_busy(self.dev, self.now) != 0 }
    }

    /// End of the run: Gemmini runs to idle; the report goes to `[gemmini] report`, else `default_report`, else
    /// stderr. Returns the cycle its busy signal last dropped.
    pub fn finish(&mut self, default_report: Option<std::path::PathBuf>) -> u64 {
        use std::ffi::CString;
        let path = self.report_path(default_report);
        let c = path.as_ref().map(|p| CString::new(p.to_string_lossy().into_owned()).unwrap());
        let done =
            unsafe { ffi::gemmini_radiance_finish(self.dev, c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) };
        if let Some(p) = &path {
            println!("Cyclotron: gemmini report in {}", p.display());
        }
        done
    }
}

#[cfg(feature = "gemmini")]
impl Drop for GemminiDevice {
    fn drop(&mut self) {
        unsafe { ffi::gemmini_radiance_free(self.dev) }
    }
}

#[cfg(not(feature = "gemmini"))]
impl GemminiDevice {
    pub fn new(_config: &GemminiConfig, _gmem: Arc<RwLock<FlatMemory>>, _smem: &mut FlatMemory) -> Self {
        panic!("[gemmini] enabled = true needs cyclotron built with `--features gemmini`");
    }
    pub fn store_ready(&mut self, _now: Cycle, _addr: u32) -> bool {
        true
    }
    pub fn store(&mut self, _addr: u32, _value: u32, _bytes: u32) {}
    pub fn load(&mut self, _addr: u32, _bytes: u32) -> u32 {
        0
    }
    pub fn busy(&mut self, _now: Cycle) -> bool {
        false
    }
    pub fn finish(&mut self, default_report: Option<std::path::PathBuf>) -> u64 {
        let _ = self.report_path(default_report);
        0
    }
}
