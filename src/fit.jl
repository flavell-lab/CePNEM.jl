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