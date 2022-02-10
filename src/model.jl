"""
Takes as input raw (non-standardized) velocity, and outputs:
- `model`: Function from parameters to noiseless model output
- `ps_0`: Initial parameters
- `ps_min`: Lower bound on parameters
- `ps_max`: Upper bound on parameters
- `xs_s`: Standardized velocity
"""
function v_model_init(raw_v)
    v = zscore(raw_v)
    xs = zeros(5, length(v))
    xs[1,:] .= raw_v
    xs[2,:] .= raw_v
    xs[3,:] .= raw_v
    xs[4,:] .= raw_v
    xs[5,:] .= raw_v
    
    xs_s = zeros(5, length(v))
    xs_s[1,:] .= v
    xs_s[2,:] .= v
    xs_s[3,:] .= v
    xs_s[4,:] .= v
    xs_s[5,:] .= v
    
    ps_0, ps_min, ps_max, list_idx_ps, list_idx_ps_reg = init_ps_model_nl6(xs, [1])
    model = generate_model_nl6_partial(xs_s, [1:800, 801:1600], list_idx_ps)
    
    return model, ps_0, ps_min, ps_max, xs_s
end

"""
Generative function for the encoder model NL6, for velocity only.

# Arguments
- `raw_v`: Raw (non-standardized) velocity
- `idx_valid`: Time points that are valid to fit
- `idx_reset_ewma`: Time points to reset EWMA (eg: at gaps between different recordings). These cannot be valid time points.
"""
@gen function v_model(raw_v::Vector{Float64}, idx_valid, idx_reset_ewma)
    # initialize parameters and zscore velocity
    (_, ps_0, ps_min, ps_max, xs_s) = v_model_init(raw_v)
    xs = xs_s[1,:]
   
    # set priors
    c1 ~ uniform(ps_min[1], ps_max[1])
    c2 ~ uniform(ps_min[2], ps_max[2])
    c3 ~ normal(ps_0[3], (ps_max[3] - ps_min[3]) / 6)
    log_λ ~ uniform(log10(ps_min[4]), log10(ps_max[4]))
    a ~ normal(0,2)
    b ~ normal(0,2)
    σ ~ exponential(10.0)
    
    λ = 10^log_λ

    # define product model without EWMA
    function model(x)
        return (sin(c1) * x + cos(c1)) * (sin(c2) * (1 - 2 * lesser(x, c3)) + cos(c2))
    end
    
    # implement EWMA
    y = Vector{Any}(undef, length(xs))
    max_t = length(xs)
    s = sum([exp(-(max_t-t)*λ) for t=1:length(xs)])
    
    prev_idx = 1
    prev_val = 0
    
    for t = 1:length(xs)
        if t in idx_reset_ewma
            @assert(!(t in idx_valid), "Index $(t) is resetting EWMA, cannot be valid.")
            y[t] = model(xs[t]) / s * a + b
        elseif t in idx_valid
            y[t] = ({(:y, t)} ~ normal(a*((y[t-1] - b) / a * (s-1) / s + model(xs[t]) / s) + b, σ))
        else
            y[t] = a*((y[t-1] - b) / a * (s-1) / s + model(xs[t]) / s) + b
        end
    end
    return (c1, c2, c3, λ, a, b, σ)
end

