module EncoderModelGen

using FlavellBase, Gen, EncoderModel, StatsBase
include("model.jl")
include("smc_fit.jl")

export
    # model.jl
    v_model_init,
    v_model,

    # smc_fit.jl
    make_constraints,
    gaussian_swap_drift_update,
    particle_filter
end