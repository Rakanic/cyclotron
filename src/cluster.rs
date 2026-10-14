use crate::base::behavior::*;
use crate::gemmini::GemminiDevice;
use crate::muon::core::MuonCore;
use crate::neutrino::neutrino::Neutrino;
use crate::sim::flat_mem::FlatMemory;
use crate::sim::log::Logger;
use crate::sim::perf_log::PerfLogSession;
use crate::sim::top::ClusterConfig;
use log::info;
use std::sync::{Arc, Mutex, RwLock};

pub struct Cluster {
    id: usize,
    pub cores: Vec<MuonCore>,
    pub neutrino: Neutrino,
    scheduled_threadblocks: usize,
    /// The cluster's MX-Gemmini ([gemmini] enabled): its registers are in shared memory, its scratchpad is it
    pub gemmini: Option<Arc<Mutex<GemminiDevice>>>,
    cycle: u64,
}

/// The cluster's Gemmini, if configured, over its shared memory; every warp of every core reaches its registers.
fn attach_gemmini(
    config: &ClusterConfig,
    gmem: &Arc<RwLock<FlatMemory>>,
    shared_mem: &Arc<RwLock<FlatMemory>>,
    cores: &mut [MuonCore],
) -> Option<Arc<Mutex<GemminiDevice>>> {
    if !config.gemmini_config.enabled {
        return None;
    }
    let mut smem = shared_mem.write().expect("shared memory lock poisoned");
    let dev = Arc::new(Mutex::new(GemminiDevice::new(
        &config.gemmini_config,
        gmem.clone(),
        &mut smem,
    )));
    for core in cores.iter_mut() {
        core.attach_gemmini(&dev);
    }
    Some(dev)
}

impl Cluster {
    pub fn new(
        config: Arc<ClusterConfig>,
        id: usize,
        logger: &Arc<Logger>,
        gmem: Arc<RwLock<FlatMemory>>,
    ) -> Self {
        let mut cores = Vec::new();
        let shared_mem = Arc::new(RwLock::new(FlatMemory::new_with_size(
            config.muon_config.smem_size,
            None,
        )));
        for cid in 0..config.muon_config.num_cores {
            cores.push(MuonCore::new(
                Arc::new(config.muon_config),
                id,
                cid, // cluster-local
                logger,
                gmem.clone(),
                shared_mem.clone(),
            ));
        }
        let gemmini = attach_gemmini(&config, &gmem, &shared_mem, &mut cores);
        Cluster {
            id,
            cores,
            neutrino: Neutrino::new(Arc::new(config.neutrino_config)),
            scheduled_threadblocks: 0,
            gemmini,
            cycle: 0,
        }
    }

    pub fn new_timed(
        config: Arc<ClusterConfig>,
        id: usize,
        logger: &Arc<Logger>,
        gmem: Arc<RwLock<FlatMemory>>,
        gmem_timing: Arc<RwLock<crate::timeflow::ClusterGmemGraph>>,
        perf_log_session: Option<Arc<PerfLogSession>>,
    ) -> Self {
        let mut cores = Vec::new();
        let shared_mem = Arc::new(RwLock::new(FlatMemory::new_with_size(
            config.muon_config.smem_size,
            None,
        )));
        for cid in 0..config.muon_config.num_cores {
            let timing_core_id = id * config.muon_config.num_cores + cid;
            cores.push(MuonCore::new_timed(
                Arc::new(config.muon_config),
                id,
                cid,
                logger,
                gmem.clone(),
                shared_mem.clone(),
                config.timing_config.clone(),
                timing_core_id,
                id,
                gmem_timing.clone(),
                perf_log_session.clone(),
            ));
        }
        let gemmini = attach_gemmini(&config, &gmem, &shared_mem, &mut cores);
        Cluster {
            id,
            cores,
            neutrino: Neutrino::new(Arc::new(config.neutrino_config)),
            scheduled_threadblocks: 0,
            gemmini,
            cycle: 0,
        }
    }

    pub fn schedule_threadblock(&mut self) {
        assert_eq!(
            self.scheduled_threadblocks, 0,
            "attempted to schedule a threadblock to an already-busy cluster. TODO: support more than one outstanding threadblocks"
        );
        info!("cluster {}: scheduled a threadblock", self.id);
        for core in &mut self.cores {
            core.spawn_single_warp();
        }
        self.scheduled_threadblocks += 1;
    }

    pub fn retired_threadblock(&self) -> usize {
        self.all_cores_retired() as usize
    }

    // TODO: This should differentiate between different threadblocks.
    pub fn all_cores_retired(&self) -> bool {
        self.cores
            .iter()
            .all(|core| core.all_warps_retired() && !core.has_timing_inflight())
    }
}

impl ModuleBehaviors for Cluster {
    fn tick_one(&mut self) {
        if let Some(g) = &self.gemmini {
            g.lock().expect("gemmini lock poisoned").set_now(self.cycle);
        }
        self.cycle += 1;
        for core in &mut self.cores {
            core.tick_one();
            core.process(&mut self.neutrino).unwrap();
        }
        self.neutrino.tick_one();
        self.neutrino
            .update(&mut self.cores.iter_mut().map(|c| &mut c.scheduler).collect());
    }

    fn reset(&mut self) {
        for core in &mut self.cores {
            core.reset();
        }
    }
}
