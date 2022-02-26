zstd = FlavellBase.standardize
logistic(x,x0,k) = 1 / (1 + exp(-k * (x - x0)))
leaky_logistic(x,x0,k,m) = logistic(x,x0,k) + m * (x - x0)
lesser(x,x0) = leaky_logistic(x0,x,50,1e-3)

const S_MULT = 10

@gen (static) function kernel_noewma(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c1::Float64), (grad)(c2::Float64), (grad)(c3::Float64), (grad)(b::Float64), σ::Float64) # latent variables
    y ~ normal(((c1+1)/sqrt(c1^2+1) - 2*c1/sqrt(c1^2+1) * lesser(xs[t], v_0)) * (c2 * xs[t] + c3) + b, σ)
    return y
end

@gen (static) function kernel_v(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c1::Float64), (grad)(c2::Float64), (grad)(c3::Float64), (grad)(s::Float64), (grad)(b::Float64), σ::Float64) # latent variables
    y ~ normal(((c1+1)/sqrt(c1^2+1) - 2*c1/sqrt(c1^2+1) * lesser(xs[t], v_0)) * (c2 * xs[t] + c3) / (S_MULT*s+1) + (y_prev - b) * S_MULT*s / (S_MULT*s+1) + b, σ * sqrt(2*S_MULT*max(s,0)+1)/(S_MULT*s+1))
    return y
end

Gen.@load_generated_functions

chain = Gen.Unfold(kernel_noewma)

chain_v = Gen.Unfold(kernel_v)

@gen (static) function unfold_v_noewma(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c1 ~ uniform(-pi/2, pi/2)
    c2 ~ normal(0,1)
    c3 ~ normal(0,1)
    b ~ normal(0,2)
    σ ~ exponential(1.0)

    chain ~ chain(t, 0.0, std_v, v_0, c1, c2, c3, b, σ)
    return 1
end

@gen (static) function unfold_v(t::Int, raw_v::Array{Float64})
    v_0 = -mean(raw_v)/std(raw_v)
    std_v = zstd(raw_v)

    c1 ~ normal(0,1)
    c2 ~ normal(0,1)
    c3 ~ normal(0,1)
    y0 ~ normal(0,1)
    s ~ exponential(10.0/S_MULT)
    b ~ normal(0,2)
    σ ~ exponential(10.0)
    
    chain ~ chain_v(t, y0, std_v, v_0, c1, c2, c3, s, b, σ)
    return 1
end

Gen.@load_generated_functions
