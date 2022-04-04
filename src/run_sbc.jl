Gen.@load_generated_functions

n_params = 9
burnin = 1000

fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1]).h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-combined_data.h5"
dict = import_data(path_h5)

n_obs = 400
model = :nl8
v = dict["velocity"][201:200+n_obs]
θh = dict["θh"][201:200+n_obs]
P = dict["pumping"][201:200+n_obs]

(trace, _) = Gen.generate(nl8, (n_obs, v, θh, P))
ys = [trace[:chain => t => :y] for t=1:n_obs]

particles_5000_2 = zeros(2047, n_params)
@time state = particle_filter_incremental(5000, v, θh, P, ys, 2, model)
output_state(state, output_path, 2047, model)

h5open(output_path, "r+") do f
    f["ground_truth"] = get_free_params(trace, model)
end

mcmc_5000 = zeros(4095, n_params)
@time traces = mcmc(xs, ys, 5095, 300)
for (i,p) in enumerate(traces)
    if i <= burnin
        continue
    end
    mcmc_5000[i-burnin,:] .= get_free_params(p)
end

h5open(output_path, "r+") do f
    f["mcmc_5000"] = mcmc_5000
end

mcmc_restart_63 = zeros(63, n_params)
@time for i=1:63
    p = mcmc(xs, ys, burnin+1, 300)[end]
    mcmc_restart_63[i,:] .= get_free_params(p)
end

h5open(output_path, "r+") do f
    f["mcmc_restart_63"] = mcmc_restart_63
end
