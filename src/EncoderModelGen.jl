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
    ℓ_MEAN,
    ℓ_STD,
    α_MEAN,
    α_STD,
    σ_RQ_MEAN,
    σ_RQ_STD,
    σ_SE_MEAN,
    σ_SE_STD,
    σ_NOISE_MEA,
    σ_NOISE_STD,
    compute_cov_matrix_vectorized_RQ,
    compute_cov_matrix_vectorized_SE,
    unfold_nl7b,
    model_nl8,
    nl8,
    nl9,
    nl10,
    nl10c,
    unfold_v_noewma,
    unfold_v,
    get_free_params,
    compute_s,
    compute_σ,

    # fit.jl
    drift_params,
    drift_α,
    drift_ℓ,
    drift_σ_RQ,
    drift_σ_SE,
    hmc_jump_update,
    particle_filter_incremental,
    output_state,
    mcmc,

    # sbc_tests.jl
    rank_test,
    χ2_uniformtest
end