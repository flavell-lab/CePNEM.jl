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
    model = generate_model_nl6_partial(xs_s, list_idx_ps, 0, [1:800, 801:1600])
    
    return model, ps_0, ps_min, ps_max, xs_s
end


function gaussian_swap_drift_update_noewma(tr)
    # Attempt to exploit symmetries
    (tr, _) = mh(tr, neg_c1_a, ())
    
    (tr, _) = mh(tr, neg_c2_a, ())
    
    (tr, _) = mh(tr, swap_c1_c2, ())
    
    # Update parameters one at a time, drifting
    (tr, _) = mh(tr, drift_proposal_a, ())
    
    (tr, _) = mh(tr, drift_proposal_b, ())
    
    (tr, _) = mh(tr, drift_proposal_c1, ())
    
    (tr, _) = mh(tr, drift_proposal_c2, ())
    
    # Update the noise parameter
    (tr, _) = mh(tr, select(:σ))
    
    # Return the updated trace
    tr
end;

@gen function drift_proposal_a(current_trace)
    a ~ normal(current_trace[:a], 0.2)
end

@gen function drift_proposal_b(current_trace)
    b ~ normal(current_trace[:b], 0.2)
end

@gen function drift_proposal_c1(current_trace)
    c1 ~ normal(current_trace[:c1], 0.1)
end

@gen function drift_proposal_c2(current_trace)
    c2 ~ normal(current_trace[:c2], 0.1)
end

@gen function neg_c1_a(current_trace)
    c1 ~ normal(-current_trace[:c1], 0.1)
    a ~ normal(-current_trace[:a], 0.1)
end

@gen function neg_c2_a(current_trace)
    c2 ~ normal(-current_trace[:c2], 0.1)
    a ~ normal(-current_trace[:a], 0.1)
end

@gen function swap_c1_c2(current_trace)
    c1 ~ normal(current_trace[:c2], 0.1)
    c2 ~ normal(current_trace[:c1], 0.1)
end
    

@gen function drift_proposal_c3(current_trace)
    c3 ~ normal(current_trace[:c3], 0.1)
end

@gen function drift_proposal_λ(current_trace)
    log_λ ~ normal(current_trace[:log_λ], 0.1)
end


@gen (static) function kernel_noewma(t::Int, y_prev::Float64, xs::Array{Float64}, c1::Float64,
        c2::Float64, c3::Float64, a::Float64, b::Float64, σ::Float64)
    y ~ normal(a * (sin(c1) * xs[t] + cos(c1)) * (sin(c2) * (1 - 2 * (xs[t] < c3)) + cos(c2)) + b, σ)
    return y
end

Gen.@load_generated_functions

chain = Gen.Unfold(kernel_noewma)

@gen (static) function unfold_v_noewma(t::Int, raw_v::Array{Float64})
    m, ps_0, ps_min, ps_max, xs_s = v_model_init(raw_v)
    xs = xs_s[1,:]
   
    c1 ~ uniform(ps_min[1], ps_max[1])
    c2 ~ uniform(ps_min[2], ps_max[2])
    c3 = ps_0[3]
    a ~ normal(0,2)
    b ~ normal(0,2)
    σ ~ exponential(10.0)
    
    chain ~ chain(t, 0.0, xs, c1, c2, c3, a, b, σ)
    return (c1, c2, c3, a, b, σ)
end

Gen.@load_generated_functions

function unfold_particle_filter_incremental(num_particles::Int, raw_v::Vector{Float64}, ys::Vector{Float64}, num_samples::Int, num_steps::Int)
    init_obs = Gen.choicemap((:chain => 1 => :y, ys[1]))
    state = Gen.initialize_particle_filter(unfold_v_noewma, (1,raw_v), init_obs, num_particles)
    for t=2:length(ys)
        if maybe_resample!(state, ess_threshold=num_particles/2)
            Threads.@threads for i=1:num_particles
                for step=1:num_steps
                    state.traces[i] = gaussian_swap_drift_update_noewma(state.traces[i])
                end
            end
        end
        obs = Gen.choicemap((:chain => t => :y, ys[t]))
        Gen.particle_filter_step!(state, (t,raw_v), (UnknownChange(),), obs)
    end
    return Gen.sample_unweighted_traces(state, num_samples)
end

function unfold_particle_filter_simultaneous(num_particles::Int, raw_v::Vector{Float64}, ys::Vector{Float64}, num_samples::Int, num_steps::Int)
    init_obs = Gen.choicemap()
    max_t = length(raw_v)
    for t=1:max_t
        init_obs[:chain => t => :y] = ys[t]
    end
    state = Gen.initialize_particle_filter(unfold_v_noewma, (max_t,raw_v), init_obs, num_particles)
    for t=1:num_steps
        Threads.@threads for i=1:num_particles
            state.traces[i] = gaussian_swap_drift_update_noewma(state.traces[i])
        end
        
        maybe_resample!(state, ess_threshold=num_particles/2)
    end
    return Gen.sample_unweighted_traces(state, num_samples)
end

function unfold_gaussian_swap_drift_inference(raw_v, ys, n_iters)
    traces = Vector{Any}(undef, n_iters)
    init_obs = Gen.choicemap()
    max_t = length(raw_v)
    for t=1:max_t
        init_obs[:chain => t => :y] = ys[t]
    end
    
    (traces[1], _) = generate(unfold_v_noewma, (max_t, raw_v), init_obs)
    for iter=2:n_iters
        traces[iter] = gaussian_swap_drift_update_noewma(traces[iter-1])
    end
    traces
end

n_params = 6
burnin = 1000
fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_output/h5/$(ARGS[1]).h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-combined_data.h5"
dict = import_data(path_h5)

xs = dict["velocity"]
n_obs = length(xs)
(trace, _) = Gen.generate(unfold_v_noewma, (n_obs, xs))
ys = [trace[:chain => t => :y] for t=1:n_obs]

particles_5000 = zeros(255, n_params)
@time particles = unfold_particle_filter_incremental(5000, xs, ys, 255, 1)
for (i,p) in enumerate(particles)
    particles_5000[i,:] .= get_retval(p)
end

particles_5000_10 = zeros(255, n_params)
@time particles = unfold_particle_filter_incremental(5000, xs, ys, 255, 10)
for (i,p) in enumerate(particles)
    particles_5000_10[i,:] .= get_retval(p)
end

particles_1000 = zeros(63, n_params)
@time particles = unfold_particle_filter_incremental(1000, xs, ys, 63, 1)
for (i,p) in enumerate(particles)
    particles_1000[i,:] .= get_retval(p)
end

particles_1000_10 = zeros(63, n_params)
@time particles = unfold_particle_filter_incremental(1000, xs, ys, 63, 10)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= get_retval(p)
end

mcmc_5000 = zeros(4095, n_params)
@time traces = unfold_gaussian_swap_drift_inference(xs, ys, 5095)
for (i,t) in enumerate(traces)
    if i <= burnin
        continue
    end
    mcmc_5000[i-burnin,:] .= get_retval(t)
end

h5open(output_path, "w") do f
    f["particles_5000"] = particles_5000
    f["particles_5000_10"] = particles_5000_10
    f["particles_1000"] = particles_1000
    f["particles_1000_10"] = particles_1000_10
    f["mcmc_5000"] = mcmc_5000
    f["ground_truth"] = collect(get_retval(trace))
end


