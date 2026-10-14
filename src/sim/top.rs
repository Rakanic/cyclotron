use crate::base::behavior::*;
use crate::base::mem::HasMemory;
use crate::cluster::Cluster;
use crate::command_proc::CommandProcessor;
use crate::gemmini::GemminiConfig;
use crate::muon::config::MuonConfig;
use crate::neutrino::config::NeutrinoConfig;
use crate::sim::config::{MemConfig, SimConfig};
use crate::sim::elf::ElfBackedMem;
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::sim::perf_log::PerfLogSession;
use crate::sim::trace_db::{default_trace_db_path, TraceDb};
use crate::timeflow::CoreGraphConfig;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

pub struct Sim {
    pub config: SimConfig,
    pub top: CyclotronTop,
    pub logger: Arc<Logger>,
    perf_log_session: Option<Arc<PerfLogSession>>,
    trace_db: Option<Mutex<TraceDb>>,
}

impl Sim {
    fn drain_traces(&mut self) {
        let Some(trace_db) = self.trace_db.as_ref() else {
            return;
        };
        let trace_db = trace_db.lock().expect("trace db lock poisoned");

        for (cluster_id, cluster) in self.top.clusters.iter_mut().enumerate() {
            for (core_id, core) in cluster.cores.iter_mut().enumerate() {
                let num_warps = core.conf().num_warps;
                for warp_id in 0..num_warps {
                    while let Some(line) = core.get_tracer_mut().consume(warp_id) {
                        trace_db.record_inst_line(cluster_id as u32, core_id as u32, &line);
                    }
                }
                while let Some(line) = core.get_mem_tracer_mut().consume() {
                    trace_db.record_mem_line(cluster_id as u32, core_id as u32, &line);
                }
            }
        }
    }

    fn write_timing_summary(&self) {
        if self.config.timing {
            let summaries = self
                .top
                .clusters
                .iter()
                .flat_map(|cluster| cluster.cores.iter().map(|core| core.timing_summary()))
                .collect::<Vec<_>>();
            if let Some(session) = &self.perf_log_session {
                session.write_summary(summaries);
            }
        }
    }

    pub fn new(
        sim_config: SimConfig,
        muon_config: MuonConfig,
        neutrino_config: NeutrinoConfig,
        mem_config: MemConfig,
    ) -> Sim {
        Self::new_with_timing(
            sim_config,
            muon_config,
            neutrino_config,
            mem_config,
            CoreGraphConfig::default(),
        )
    }

    pub fn new_with_timing(
        sim_config: SimConfig,
        muon_config: MuonConfig,
        neutrino_config: NeutrinoConfig,
        mem_config: MemConfig,
        timing_config: CoreGraphConfig,
    ) -> Sim {
        Self::new_with_gemmini(
            sim_config,
            muon_config,
            neutrino_config,
            mem_config,
            timing_config,
            GemminiConfig::default(),
        )
    }

    pub fn new_with_gemmini(
        sim_config: SimConfig,
        muon_config: MuonConfig,
        neutrino_config: NeutrinoConfig,
        mem_config: MemConfig,
        timing_config: CoreGraphConfig,
        gemmini_config: GemminiConfig,
    ) -> Sim {
        let perf_log_session = if sim_config.timing {
            PerfLogSession::new().map(Arc::new)
        } else {
            None
        };
        let trace_db = if sim_config.trace {
            let trace_db_path = default_trace_db_path(None, Some(sim_config.elf.as_path()));
            println!(
                "Cyclotron: writing ISA instruction trace to {}",
                trace_db_path.display()
            );
            Some(Mutex::new(TraceDb::new(&trace_db_path)))
        } else {
            None
        };
        let logger = Arc::new(Logger::new(sim_config.log_level));
        let top = CyclotronTop::new(
            Arc::new(CyclotronConfig {
                timeout: sim_config.timeout,
                elf: sim_config.elf.clone(),
                cluster_config: ClusterConfig {
                    muon_config,
                    neutrino_config,
                    timing_config,
                    gemmini_config,
                },
                mem_config,
                timing_enabled: sim_config.timing,
            }),
            &logger,
            perf_log_session.clone(),
        );

        let mut sim = Sim {
            config: sim_config,
            top,
            logger,
            perf_log_session,
            trace_db,
        };
        sim.top.reset();
        sim
    }

    /// Each cluster's Gemmini runs to idle and reports (gemmini_report.txt in the perf log directory, by default).
    fn finish_gemmini(&mut self) {
        let dir = self.perf_log_session.as_ref().map(|s| s.run_dir().to_path_buf());
        for (id, cluster) in self.top.clusters.iter().enumerate() {
            if let Some(g) = &cluster.gemmini {
                let mut g = g.lock().expect("gemmini lock poisoned");
                let path = dir.as_ref().map(|d| d.join(format!("gemmini_report_cluster{}.txt", id)));
                let done = g.finish(path);
                println!(
                    "Cyclotron: cluster {} gemmini: {} commands ({} INST stores waited), busy until cycle {}",
                    id, g.commands, g.refused, done
                );
            }
        }
    }

    fn smem_init(&mut self) {
        for cluster in &self.top.clusters {
            let mut smem = cluster.cores[0].shared_mem.write().expect("shared memory lock poisoned");
            for &[addr, value] in &self.config.smem_init {
                smem.write(addr as usize, &value.to_le_bytes()).expect("[sim] smem_init write failed");
            }
        }
    }

    fn smem_dump(&self) {
        for (id, cluster) in self.top.clusters.iter().enumerate() {
            let smem = cluster.cores[0].shared_mem.read().expect("shared memory lock poisoned");
            for &[addr, count] in &self.config.smem_dump {
                let words = (0..count)
                    .map(|i| {
                        let w = smem.read_n::<4>((addr + 4 * i) as usize).expect("[sim] smem_dump read failed");
                        format!("{:08x}", u32::from_le_bytes(w))
                    })
                    .collect::<Vec<_>>();
                println!("Cyclotron: cluster {} smem 0x{:x}: {}", id, addr, words.join(" "));
            }
        }
    }

    pub fn simulate(&mut self) -> Result<(), u32> {
        self.top.reset();
        self.smem_init();
        for cycle in 0..self.top.timeout {
            if self.top.finished() {
                println!("simulation finished after {} cycles", cycle + 1);
                self.write_timing_summary();
                self.finish_gemmini();
                self.smem_dump();
                return self.check_tohost();
            }
            self.tick();
        }

        self.write_timing_summary();
        self.finish_gemmini();
        self.smem_dump();

        Err(0)
    }

    pub fn check_tohost(&self) -> Result<(), u32> {
        if let Some(tohost) = self.top.clusters[0].cores[0].scheduler.tohost() {
            if tohost != 0 {
                let case = tohost >> 1;
                println!(
                    "Cyclotron: isa-test failed with tohost={}, case={}",
                    tohost, case
                );
                return Err(tohost);
            } else {
                println!("Cyclotron: isa-test passed with tohost={}", tohost);
            }
        }
        return Ok(());
    }

    /// Advances all cores by one instruction.
    pub fn tick(&mut self) {
        if self.top.finished() {
            return;
        }
        self.top.tick_one();
        self.drain_traces();
    }

    pub fn finished(&self) -> bool {
        self.top.finished()
    }
}

pub struct CyclotronConfig {
    pub timeout: u64, // TODO: use sim
    pub elf: PathBuf, // TODO: use sim
    pub cluster_config: ClusterConfig,
    pub mem_config: MemConfig,
    pub timing_enabled: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ClusterConfig {
    pub muon_config: MuonConfig,
    pub neutrino_config: NeutrinoConfig,
    pub timing_config: CoreGraphConfig,
    #[serde(default)]
    pub gemmini_config: GemminiConfig,
}

pub struct CyclotronTop {
    pub cproc: CommandProcessor,
    pub clusters: Vec<Cluster>,
    pub timeout: u64,
    pub gmem: Arc<RwLock<FlatMemory>>,
}

impl CyclotronTop {
    pub fn new(
        config: Arc<CyclotronConfig>,
        logger: &Arc<Logger>,
        perf_log_session: Option<Arc<PerfLogSession>>,
    ) -> CyclotronTop {
        let elf_path = Path::new(&config.elf);
        let imem = ElfBackedMem::new(&elf_path);
        let mut clusters = Vec::new();
        let cluster_config = Arc::new(config.cluster_config.clone());

        // TODO: current implementation means imem is writable, but this is true
        // in hardware too?
        let mut gmem = FlatMemory::new(Some(config.mem_config));
        gmem.copy_elf(&imem);

        let gmem = Arc::new(RwLock::new(gmem));

        // TODO: parameterize
        let num_clusters = 1;
        let cores_per_cluster = cluster_config.muon_config.num_cores.max(1);
        if config.timing_enabled {
            let gmem_timing = Arc::new(RwLock::new(crate::timeflow::ClusterGmemGraph::new(
                cluster_config.timing_config.memory.gmem.clone(),
                num_clusters,
                cores_per_cluster,
            )));
            for id in 0..num_clusters {
                clusters.push(Cluster::new_timed(
                    cluster_config.clone(),
                    id,
                    logger,
                    gmem.clone(),
                    gmem_timing.clone(),
                    perf_log_session.clone(),
                ));
            }
        } else {
            for id in 0..num_clusters {
                clusters.push(Cluster::new(
                    cluster_config.clone(),
                    id,
                    logger,
                    gmem.clone(),
                ));
            }
        }
        CyclotronTop {
            cproc: CommandProcessor::new(
                Arc::new(config.cluster_config.muon_config.clone()),
                1, /*FIXME: properly get thread dimension*/
            ),
            clusters,
            timeout: config.timeout,
            gmem,
        }
    }

    pub fn schedule_clusters(&mut self) {
        let tb_schedule = self.cproc.schedule();
        let num_clusters = self.clusters.len();
        for id in 0..num_clusters {
            if tb_schedule[id] {
                self.clusters[id].schedule_threadblock();
            }
        }
    }

    /// Load a word from gmem.
    pub fn gmem_load(&self, addr: u32) -> [u8; 4] {
        self.gmem
            .read()
            .expect("lock poisoned")
            .read_n::<4>(addr as usize)
            .expect("load failed")
    }

    /// Store a data with given size to gmem.  Doesn't support partial writes.
    pub fn gmem_store(&self, addr: u32, data: u32, size: u32) {
        let mut gmem = self.gmem.write().expect("lock poisoned");
        let data_bytes = data.to_le_bytes();
        match size {
            0 => gmem.write(addr as usize, &data_bytes[0..1]), // store byte
            1 => gmem.write(addr as usize, &data_bytes[0..2]), // store half
            2 => gmem.write(addr as usize, &data_bytes[0..4]), // store word
            _ => panic!("unimplemented store type"),
        }
        .expect("store failed");
    }

    pub fn finished(&self) -> bool {
        self.clusters.iter().all(|cl| cl.all_cores_retired())
    }
}

impl ModuleBehaviors for CyclotronTop {
    fn tick_one(&mut self) {
        self.schedule_clusters();

        self.clusters.iter_mut().for_each(Cluster::tick_one);

        // FIXME: if tick_one() is called after finished() is true, the retire() call might result
        // in an underflow.
        let num_clusters = self.clusters.len();
        for id in 0..num_clusters {
            let retired_tb = self.clusters[id].retired_threadblock();
            self.cproc.retire(id, retired_tb);
        }
    }

    fn reset(&mut self) {
        self.clusters.iter_mut().for_each(Cluster::reset);
    }
}
