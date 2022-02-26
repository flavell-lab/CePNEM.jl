Gen.@load_generated_functions

n_params = 7
burnin = 500
fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_output_ewma/h5/$(ARGS[1]).h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-combined_data.h5"
dict = import_data(path_h5)

n_obs = 300
xs = dict["velocity"][1:n_obs]
(trace, _) = Gen.generate(unfold_v, (n_obs, xs))
ys = [trace[:chain => t => :y] for t=1:n_obs]

h5open(output_path, "w") do f
    f["ground_truth"] = get_params(trace)
end

particles_5000 = zeros(255, n_params)
@time particles = particle_filter_incremental(5000, xs, ys, 255, 1)
for (i,p) in enumerate(particles)
    particles_5000[i,:] .= get_params(p)
end

h5open(output_path, "r+") do f
    f["particles_5000"] = particles_5000
end


particles_1000 = zeros(63, n_params)
@time particles = particle_filter_incremental(1000, xs, ys, 63, 1)
for (i,p) in enumerate(particles)
    particles_1000[i,:] .= get_params(p)
end

h5open(output_path, "r+") do f
    f["particles_1000"] = particles_1000
end

particles_1000_10 = zeros(63, n_params)
@time particles = particle_filter_incremental(1000, xs, ys, 63, 10)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= get_params(p)
end

h5open(output_path, "r+") do f
    f["particles_1000_10"] = particles_1000_10
end

mcmc_5000 = zeros(4095, n_params)
@time traces = mcmc(xs, ys, 4595, 300)
for (i,p) in enumerate(traces)
    if i <= burnin
        continue
    end
    mcmc_5000[i-burnin,:] .= get_params(p)
end

h5open(output_path, "r+") do f
    f["mcmc_5000"] = mcmc_5000
end

mcmc_restart_63 = zeros(63, n_params)
@time for i=1:63
    p = mcmc(xs, ys, burnin+1, 300)[end]
    mcmc_restart_63[i,:] .= get_params(p)
end

h5open(output_path, "r+") do f
    f["mcmc_restart_63"] = mcmc_restart_63
end

