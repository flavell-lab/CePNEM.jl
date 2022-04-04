Gen.@load_generated_functions

n_params = 9

fit_uid = "2021-05-26-07"
output_path_gt = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1])_gt.h5"
output_path_smc5000_1 = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1])_smc5000_1.h5"
output_path_smc5000_2 = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1])_smc5000_2.h5"
output_path_mcmc = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1])_mcmc.h5"
# output_path_mcmc_init = "/om2/user/aaatanas/gen_sbc_nl8/h5/$(ARGS[1])_mcmc_init.h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-data.h5"
dict = import_data(path_h5)

n_obs = 400
model = :nl8
v = dict["velocity"][201:200+n_obs]
θh = dict["θh"][201:200+n_obs]
P = dict["pumping"][201:200+n_obs]

(trace, _) = Gen.generate(nl8, (n_obs, v, θh, P))
ys = [trace[:chain => t => :y] for t=1:n_obs]


h5open(output_path_gt, "w") do f
    f["ground_truth"] = get_free_params(trace, model)
end

particles_5000_2 = zeros(2047, n_params)
@time state = particle_filter_incremental(5000, v, θh, P, ys, 1, model)
output_state(state, output_path_smc5000_1, 2047, model)


particles_5000 = zeros(2047, n_params)
@time state = particle_filter_incremental(5000, v, θh, P, ys, 2, model)
output_state(state, output_path_smc5000_2, 2047, model)

mcmc_5000 = zeros(4095, n_params)
@time traces = mcmc(xs, ys, 5095, 300)
for (i,p) in enumerate(traces)
    mcmc_5000[i,:] .= get_free_params(p)
end

h5open(output_path_mcmc, "w") do f
    f["mcmc"] = mcmc_5000
end
