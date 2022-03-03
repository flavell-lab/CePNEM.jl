Gen.@load_generated_functions

n_params = 7
fit_uid = "2021-05-26-07"
output_path = "/om2/user/aaatanas/gen_output_nl7b/h5/$(ARGS[1]).h5"
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

particles_5000 = zeros(2047, n_params)
@time particles = particle_filter_incremental(5000, v, θh, P, ys, 2047, 1, model)
for (i,p) in enumerate(particles)
    particles_5000[i,:] .= get_free_params(p, model)
end

h5open(output_path, "r+") do f
    f["particles_5000"] = particles_5000
end

particles_1000 = zeros(511, n_params)
@time particles = particle_filter_incremental(1000, v, θh, P, ys, 511, 1, model)
for (i,p) in enumerate(particles)
    particles_1000[i,:] .= get_free_params(p, model)
end

h5open(output_path, "r+") do f
    f["particles_1000"] = particles_1000
end

particles_1000_10 = zeros(511, n_params)
@time particles = particle_filter_incremental(1000, v, θh, P, ys, 511, 10, model)
for (i,p) in enumerate(particles)
    particles_1000_10[i,:] .= get_free_params(p, model)
end

h5open(output_path, "r+") do f
    f["particles_1000_10"] = particles_1000_10
end


