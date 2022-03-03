Gen.@load_generated_functions

n_params = 9
fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_output_nl7b_2/h5/$(ARGS[1]).h5"
path_h5 = "/om2/user/aaatanas/processed_h5/$(fit_uid)-combined_data.h5"
dict = import_data(path_h5)

n_obs = 400
model = :nl7b
v = dict["velocity"][201:200+n_obs]
θh = dict["θh"][201:200+n_obs]
P = dict["pumping"][201:200+n_obs]


(trace, _) = Gen.generate(unfold_nl7b, (n_obs, v, θh, P))
ys = [trace[:chain => t => :y] for t=1:n_obs]

h5open(output_path, "w") do f
    f["ground_truth"] = get_free_params(trace, model)
end

particles_5000_2 = zeros(2047, n_params)
@time particles = particle_filter_incremental(5000, v, θh, P, ys, 2047, 2, model)
for (i,p) in enumerate(particles)
    particles_5000_2[i,:] .= get_free_params(p, model)
end

h5open(output_path, "r+") do f
    f["particles_5000_2"] = particles_5000_2
end

particles_2000_10 = zeros(1023, n_params)
@time particles = particle_filter_incremental(2000, v, θh, P, ys, 1023, 10, model)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= get_free_params(p, model)
end

h5open(output_path, "r+") do f
    f["particles_2000_10"] = particles_2000_10
end
