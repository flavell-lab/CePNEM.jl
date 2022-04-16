zstd = FlavellBase.standardize
logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x - x0)
lesser(x,x0) = leaky_logistic(x0,x,50,1e-3)

const s_MEAN = 10
const σ_MEAN = 0.25
const v_STD = 0.06030961137253011
const vT_STD = 0.9425527026496543
const θh_STD = 0.49429038957075727
const P_STD = 1.2772001409506841

@gen (static) function kernel_noewma(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c::Float64), (grad)(b::Float64), σ::Float64) # latent variables
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(xs[t], v_0)) * (c_v * xs[t] + c) + b, σ)
    return y
end

@gen (static) function kernel_v(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c::Float64), (grad)(s::Float64), (grad)(b::Float64), (grad)(σ::Float64)) # latent variables
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(xs[t], v_0)) * (c_v * xs[t] + c) / (s+1) + (y_prev - b) * s / (s+1) + b, σ)
    return y
end

@gen (grad, static) function kernel_nl7b(t::Int, (grad)(y_prev::Float64), std_v::Array{Float64}, 
            std_θh::Array{Float64}, std_P::Array{Float64}, v_0::Float64,
            (grad)(c_vT::Float64), (grad)(c_v::Float64), (grad)(c_θh::Float64),
            (grad)(c_P::Float64), (grad)(c::Float64),
            (grad)(s::Float64), (grad)(b::Float64), (grad)(σ::Float64))
    y ~ normal(((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * lesser(std_v[t], v_0)) * 
            (c_v * std_v[t] + c_θh * std_θh[t] + c_P * std_P[t] + c) / (s+1)
            + (y_prev - b) * s / (s+1) + b, σ)
    return y
end

function model_nl8(max_t::Int, c_vT::T, c_v::T, c_θh::T, c_P::T, c::T, y0::T, s0::T, b::T, v::Vector{Float64}, θh::Vector{Float64}, P::Vector{Float64}) where T
    std_v = v ./ v_STD
    std_θh = θh ./ θh_STD
    std_P = P ./ P_STD
    
    activity = zeros(T, max_t)
    s = compute_s(s0)
    activity_prev = y0

    for t=1:max_t
        activity[t] = ((c_vT+1)/sqrt(c_vT^2+1) - 2*c_vT/sqrt(c_vT^2+1) * (std_v[t] < 0)) * 
                (c_v * std_v[t] + c_θh * std_θh[t] + c_P * std_P[t] + c) / (s+1) + (activity_prev - b) * s / (s+1) + b
        activity_prev = activity[t]
    end
    return activity
end

@gen (grad, static) function kernel_nl8((grad)(z::Float64), (grad)(σ::Float64))
    y ~ normal(z, σ)
    return y
end

Gen.@load_generated_functions

chain = Gen.Unfold(kernel_noewma)

chain_v = Gen.Unfold(kernel_v)

chain_nl7b = Gen.Unfold(kernel_nl7b)

chain_nl8 = Gen.Map(kernel_nl8)


@gen (static) function unfold_nl7b(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0
    std_v = v/v_STD
    std_θh = θh/θh_STD
    std_P = P/P_STD

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0 ~ normal(0,1)
    
    s = compute_s(s0)
    σ = compute_σ(σ0)

    chain ~ chain_nl7b(t, y0, std_v, std_θh, std_P, v_0, c_vT, c_v, c_θh, c_P, c, s, b, σ)
    return 1
end

@gen function nl8(max_t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0 ~ normal(0,1)

    σ = compute_σ(σ0)
    
    z = model_nl8(max_t, c_vT, c_v, c_θh, c_P, c, y0, s0, b, v[1:max_t], θh[1:max_t], P[1:max_t])
    chain ~ chain_nl8(z, fill(σ, max_t))
    
    return 1
end

@gen (static) function nl9(t::Int, v::Array{Float64}, θh::Array{Float64}, P::Array{Float64})
    v_0 = 0.0
    std_v = v/v_STD
    std_θh = θh/θh_STD
    std_P = P/P_STD

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c_θh ~ normal(0,1)
    c_P ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,1)
    σ0_model ~ normal(0,1)
    σ0_measure ~ normal(0,1)
    
    s = compute_s(s0)
    σ_model = compute_σ(σ0_model)
    σ_measure = compute_σ(σ0_measure)

    chain_model ~ chain_nl7b(t, y0, std_v, std_θh, std_P, v_0, c_vT, c_v, c_θh, c_P, c, s, b, σ_model)
    chain ~ chain_nl8(chain_model, fill(σ_measure, t))
    return 1
end

@gen (static) function unfold_v_noewma(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c_vT ~ uniform(-pi/2, pi/2)
    c_v ~ normal(0,1)
    c ~ normal(0,1)
    b ~ normal(0,2)
    σ ~ exponential(1.0)

    chain ~ chain(t, 0.0, std_v, v_0, c_vT, c_v, c, b, σ)
    return 1
end

@gen (static) function unfold_v(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c_vT ~ normal(0,1)
    c_v ~ normal(0,1)
    c ~ normal(0,1)
    y0 ~ normal(0,1)
    s0 ~ normal(0,1)
    b ~ normal(0,2)
    σ0 ~ normal(0,1)
    
    s = compute_s(s0)
    σ = compute_σ(σ0)

    chain ~ chain_v(t, y0, std_v, v_0, c_vT, c_v, c, s, b, σ)
    return 1
end

function compute_s(s0)
    return s_MEAN * exp(s0)
end

function compute_σ(σ0)
    return σ_MEAN * exp(σ0/2)
end

Gen.@load_generated_functions

function get_free_params(trace, model)
    if model == :nl9
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0_model], trace[:σ0_measure]]
    elseif model in [:nl7b, :nl8]
        return [trace[:c_vT], trace[:c_v], trace[:c_θh], trace[:c_P], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0]]
    elseif model == :v
        return [trace[:c_vT], trace[:c_v], trace[:c], trace[:y0], trace[:s0], trace[:b], trace[:σ0]]
    elseif model == :v_noewma
        return [trace[:c_vT], trace[:c_v], trace[:c], trace[:b], trace[:σ]]
    end
end
