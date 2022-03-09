module EncoderModelGen

using Gen
using FlavellBase
using HDF5
using Statistics
using StatsBase
using Distributions

include("fit.jl")
include("model.jl")
include("data.jl")
include("sbc_tests.jl")

export
    # data.jl
    import_data,

    # model.jl
    s_MEAN,
    σ_MEAN,
    v_STD,
    vT_STD,
    θh_STD,
    P_STD,
    unfold_nl7b,
    unfold_v_noewma,
    unfold_v,
    get_free_params,
    compute_s,
    compute_σ,

    # fit.jl
    hmc_jump_update,
    particle_filter_incremental,
    output_state,
    mcmc,

    # sbc_tests.jl
    rank_test,
    χ2_uniformtest
end