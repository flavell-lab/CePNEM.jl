"""
Generates a choicemap that sets values of the data to `ys` at time points in `idx_valid`.
"""
function make_constraints(ys::Vector{Float64}, idx_valid)
    constraints = Gen.choicemap()
    for i=idx_valid
        constraints[(:y, i)] = ys[i]
    end
    return constraints
end

"""
Proposes many different MH steps, which either expolit symmetries of the model, or simply propose Gaussian drift.
"""
function gaussian_swap_drift_update(tr)
    # Attempt to exploit symmetries
    (tr, _) = mh(tr, neg_c1_a, ())
    
    (tr, _) = mh(tr, neg_c2_a, ())
    
    (tr, _) = mh(tr, swap_c1_c2, ())
    
    # Update parameters one at a time, drifting
    (tr, _) = mh(tr, drift_proposal_a, ())
    
    (tr, _) = mh(tr, drift_proposal_b, ())
    
    (tr, _) = mh(tr, drift_proposal_c1, ())
    
    (tr, _) = mh(tr, drift_proposal_c2, ())
    
    (tr, _) = mh(tr, drift_proposal_c3, ())
    
    (tr, _) = mh(tr, drift_proposal_λ, ())
    
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

"""
Implements a particle filter with MCMC rejuvenation. Each step of the particle filter attempts to directly estimate the posterior.

# Arguments:
- `model`: Generative model to fit
- `args`: Arguments to the model, as a tuple
- `ys::Vector{Float64}`: Observations (ie: neural trace)
- `num_particles::Int`: Number of particles to use
- `num_samples::Int`: Number of samples to draw from final particle distribution
- `num_steps::Int`: Number of steps to update particles.
"""
function particle_filter(model, args, ys::Vector{Float64}, num_particles::Int, num_samples::Int, num_steps::Int)
    # construct initial observations
    init_obs = make_constraints(ys, 1:length(ys))
    state = Gen.initialize_particle_filter(model, args, init_obs, num_particles)

    # steps
    for t=1:num_steps
        # MCMC rejuvenate
        Threads.@threads for i=1:num_particles
            state.traces[i] = gaussian_swap_drift_update(state.traces[i])
        end
        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
    end

    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end
