using Gen
using EncoderModel
using FlavellBase
using HDF5
using Statistics
using StatsBase

zstd = FlavellBase.standardize

function import_data(path_h5)
    dict_ = Dict{String,Any}()
    
    h5open(path_h5,"r") do h5f
        velocity = read(h5f, "behavior/velocity")
        reversal_vec = read(h5f, "behavior/reversal_vec")
        reversal_events = read(h5f, "behavior/reversal_events")
        head_angle = read(h5f, "behavior/head_angle")
        head_angle_derivative = read(h5f, "behavior/head_angle_derivative")
        angular_velocity = read(h5f, "behavior/angular_velocity")
        pumping = read(h5f, "behavior/pumping")
        worm_curvature = read(h5f, "behavior/worm_curvature")
        trace = read(h5f, "gcamp/trace_array")
        list_splits = read(h5f, "gcamp/idx_splits")
        
        dict_["idx_splits"] = [list_splits[i,1]:list_splits[i,2] for i = 1:size(list_splits,1)]
        dict_["trace_array"] = trace
        dict_["n_neuron"] = size(trace, 1)
        dict_["n_t"] = size(trace, 2)
        dict_["idx_splits"] = [list_splits[i,1]:list_splits[i,2] for i = 1:size(list_splits,1)]
        dict_["velocity"] = velocity
        dict_["θh"] = head_angle
        dict_["dθh"] = head_angle_derivative
        dict_["dorsal"] = max.(0, head_angle)
        dict_["ventral"] = max.(0, -head_angle)
        dict_["ang_vel"] = angular_velocity
        dict_["fwd_d"] = (1 .- reversal_vec) .* dict_["dorsal"]
        dict_["fwd_v"] = (1 .- reversal_vec) .* dict_["ventral"]
        dict_["rev_d"] = reversal_vec .* dict_["dorsal"]
        dict_["rev_v"] = reversal_vec .* dict_["ventral"]
        dict_["curve"] = worm_curvature
        dict_["speed_reversal"] = max.(-velocity,0)
        dict_["pumping"] = pumping
        
        for var = ["θh", "dθh", "dorsal", "ventral", "ang_vel",
            "fwd_d", "fwd_v", "rev_d", "rev_v", "curve", "velocity", "pumping"]
            dict_[var*"_s"] = zstd(dict_[var])
            dict_["s_"*var] = std(dict_[var])
            dict_["u_"*var] = mean(dict_[var])
        end
    end
    
    dict_
end


"""
In case of c2 near 0, propose massive jump in c1 and shift all other variables accordingly
"""
@gen function jump_c1(current_trace, neg)
    vT_coeff = -2 * current_trace[:c1] * current_trace[:c3] / sqrt(current_trace[:c1]^2 + 1)
    c_coeff = (1+current_trace[:c1]) * current_trace[:c3] / sqrt(current_trace[:c1]^2+1) + current_trace[:b]
    c1 ~ normal(current_trace[:c1] * neg,1)
    c3 ~ normal(vT_coeff/(-2*c1/sqrt(c1^2+1)), 1e-4)
    b ~ normal(c_coeff - (1+c1)*c3/sqrt(c1^2+1), 1e-4)
end


"""
In case of c1 near 0, propose massive jump in c3 to the prior and shift all other variables accordingly
"""
@gen function jump_c3(current_trace, neg)
    c_coeff = (1+current_trace[:c1]) * current_trace[:c3] / sqrt(current_trace[:c1]^2+1) + current_trace[:b]
    c3 ~ normal(current_trace[:c3] * neg, 1)
    b ~ normal(c_coeff - (1+current_trace[:c1])*c3/sqrt(current_trace[:c1]^2+1), 1e-4)
end

"""
Use the fact that vT = (v < 0) and v are similar to propose effectively swapping the two variables
Specifically, moves parameter values along the manifold where v = (μ_vT - vT) / σ_vT. 
"""
@gen function jump_all(current_trace, neg_c1, neg_c2, μ_vT, σ_vT)
    curr_c1 = current_trace[:c1]
    curr_c2 = current_trace[:c2]
    curr_c3 = current_trace[:c3]
    vT_coeff = (-2*curr_c1*curr_c3*σ_vT + 2*curr_c1*curr_c2 - (1+curr_c1)*curr_c2 - 2*curr_c1*curr_c2*μ_vT) / (sqrt(curr_c1^2+1) * σ_vT)
    c_coeff = current_trace[:b] + ((1+curr_c1)*curr_c3) / sqrt(curr_c1^2+1) + ((1+curr_c1)*curr_c2*μ_vT)/(sqrt(curr_c1^2+1)*σ_vT)
    
    c1 ~ normal(current_trace[:c1] * neg_c1, 1)
    c2 ~ normal(current_trace[:c2] * neg_c2, 1)
    c3 ~ normal((-c2 + c1*c2 - 2*c1*c2*μ_vT - sqrt(c1^2+1)*σ_vT*vT_coeff)/(2*c1*σ_vT), 1e-4)
    b ~ normal(c_coeff - ((1+c1)*c3+(1+c1)*c2*μ_vT/σ_vT)/sqrt(c1^2+1), 1e-4)
end

function hmc_jump_update(tr, μ_vT, σ_vT)
    # apply HMC to the entire trace
    (tr, accept) = hmc(tr, select(:c1, :c2, :c3, :b), eps=tr[:σ]/20)
    
    # apply "jump" transforms that attempt to exploit symmetries of the kernel
    neg = rand([-1, 1])
    (tr, accept) = mh(tr, jump_c1, (neg,))   
    neg = rand([-1, 1])
    (tr, accept) = mh(tr, jump_c3, (neg,))   
    neg_c1 = rand([-1,1])
    neg_c2 = rand([-1,1])
    (tr, accept) = mh(tr, jump_all, (neg_c1, neg_c2, μ_vT, σ_vT))
    
    # update noise value
    (tr, accept) = mh(tr, select(:σ))
    
    return tr
end

@gen (static) function kernel_noewma(t::Int, y_prev::Float64, xs::Array{Float64}, v_0::Float64,
        (grad)(c1::Float64), (grad)(c2::Float64), (grad)(c3::Float64), (grad)(b::Float64), σ::Float64) # latent variables
    y ~ normal(((c1+1)/sqrt(c1^2+1) - 2*c1/sqrt(c1^2+1) * lesser(xs[t], v_0)) * (c2 * xs[t] + c3) + b, σ)
    return y
end

Gen.@load_generated_functions

chain = Gen.Unfold(kernel_noewma)

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

Gen.@load_generated_functions

function particle_filter_incremental(num_particles::Int, raw_v::Vector{Float64}, ys::Vector{Float64}, num_samples::Int, num_steps::Int)
    μ_vT = mean(raw_v .< 0)
    σ_vT = std(raw_v .< 0)
    init_obs = Gen.choicemap((:chain => 1 => :y, ys[1]))
    state = Gen.initialize_particle_filter(unfold_v_noewma, (1,raw_v), init_obs, num_particles)
    for t=2:length(ys)
        if maybe_resample!(state, ess_threshold=num_particles/2)
            for i=1:num_particles
                for step=1:num_steps
                    state.traces[i] = hmc_jump_update(state.traces[i], μ_vT, σ_vT)
                end
            end
        end
        obs = Gen.choicemap((:chain => t => :y, ys[t]))
        Gen.particle_filter_step!(state, (t,raw_v), (IntDiff(1), NoChange()), obs)
    end
    return Gen.sample_unweighted_traces(state, num_samples)
end

function mcmc(raw_v, ys, n_iters, max_t)
    μ_vT = mean(raw_v .< 0)
    σ_vT = std(raw_v .< 0)
    traces = Vector{Any}(undef, n_iters)
    init_obs = Gen.choicemap()
    for t=1:max_t
        init_obs[:chain => t => :y] = ys[t]
    end
    
    (traces[1], _) = generate(unfold_v_noewma, (max_t, raw_v), init_obs)
    for iter=2:n_iters
        traces[iter] = hmc_jump_update(traces[iter-1], μ_vT, σ_vT)
    end
    traces
end

n_params = 5
burnin = 500
fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_output_2/h5/$(ARGS[1]).h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-combined_data.h5"
dict = import_data(path_h5)

n_obs = 300
xs = dict["velocity"][1:n_obs]
(trace, _) = Gen.generate(unfold_v_noewma, (n_obs, xs))
ys = [trace[:chain => t => :y] for t=1:n_obs]

h5open(output_path, "w") do f
    f["ground_truth"] = [trace[:c1], trace[:c2], trace[:c3], trace[:b], trace[:σ]]
end

particles_5000 = zeros(255, n_params)
@time particles = particle_filter_incremental(5000, xs, ys, 255, 1)
for (i,p) in enumerate(particles)
    particles_5000[i,:] .= [p[:c1], p[:c2], p[:c3], p[:b], p[:σ]]
end

h5open(output_path, "r+") do f
    f["particles_5000"] = particles_5000
end

particles_5000_10 = zeros(255, n_params)
@time particles = particle_filter_incremental(5000, xs, ys, 255, 10)
for (i,p) in enumerate(particles)
    particles_5000_10[i,:] .= [p[:c1], p[:c2], p[:c3], p[:b], p[:σ]]
end

h5open(output_path, "r+") do f
    f["particles_5000_10"] = particles_5000_10
end

particles_1000 = zeros(63, n_params)
@time particles = particle_filter_incremental(1000, xs, ys, 63, 1)
for (i,p) in enumerate(particles)
    particles_1000[i,:] .= [p[:c1], p[:c2], p[:c3], p[:b], p[:σ]]
end

h5open(output_path, "r+") do f
    f["particles_1000"] = particles_1000
end

particles_1000_10 = zeros(63, n_params)
@time particles = particle_filter_incremental(1000, xs, ys, 63, 10)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= [p[:c1], p[:c2], p[:c3], p[:b], p[:σ]]
end

h5open(output_path, "r+") do f
    f["particles_1000_10"] = particles_1000_10
end

particles_1000_50 = zeros(63, n_params)
@time particles = particle_filter_incremental(1000, xs, ys, 63, 50)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= [p[:c1], p[:c2], p[:c3], p[:b], p[:σ]]
end

h5open(output_path, "r+") do f
    f["particles_1000_50"] = particles_1000_10
end

mcmc_5000 = zeros(4095, n_params)
@time traces = mcmc(xs, ys, 4595, 300)
for (i,t) in enumerate(traces)
    if i <= burnin
        continue
    end
    mcmc_5000[i-burnin,:] .= [t[:c1], t[:c2], t[:c3], t[:b], t[:σ]]
end

h5open(output_path, "r+") do f
    f["mcmc_5000"] = mcmc_5000
end

mcmc_restart_63 = zeros(63, n_params)
@time for i=1:63
    t = mcmc(xs, ys, burnin+1, 300)[end]
    mcmc_restart_63[i,:] .= [t[:c1], t[:c2], t[:c3], t[:b], t[:σ]]
end

h5open(output_path, "r+") do f
    f["mcmc_restart_63"] = mcmc_restart_63
end
