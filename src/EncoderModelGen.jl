module EncoderModelGen

using Gen
using FlavellBase
using HDF5
using Statistics
using StatsBase

include("fit.jl")
include("model.jl")
include("data.jl")

export
    # data.jl
    import_data,

    # model.jl
    unfold_v_noewma,
    unfold_v,
    S_STD,
    get_free_params,

    # fit.jl
    hmc_jump_update,
    hmc_jump_update_noewma,
    particle_filter_incremental,
    mcmc
end