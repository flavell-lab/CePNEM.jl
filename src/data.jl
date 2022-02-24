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