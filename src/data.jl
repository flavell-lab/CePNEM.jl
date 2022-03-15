function import_data(path_h5)
    dict_ = Dict{String,Any}()
    
    h5open(path_h5,"r") do h5f
        # behavior
        velocity = read(h5f, "behavior/velocity")
        reversal_vec = read(h5f, "behavior/reversal_vec")
        reversal_events = read(h5f, "behavior/reversal_events")
        head_angle = read(h5f, "behavior/head_angle")
        angular_velocity = read(h5f, "behavior/angular_velocity")
        pumping = read(h5f, "behavior/pumping")
        worm_curvature = read(h5f, "behavior/worm_curvature")
        body_angle_absolute = read(h5f, "behavior/body_angle_absolute")
        body_angle_all = read(h5f, "behavior/body_angle_all")
        body_angle = read(h5f, "behavior/body_angle")

        # trace
        trace_F20 = read(h5f, "gcamp/traces_array_F_F20")
        trace_original = read(h5f, "gcamp/trace_array_original")
        trace = read(h5f, "gcamp/trace_array")
        list_splits = read(h5f, "gcamp/idx_splits")
        
        ## put into dictionary
        # trace
        dict_["idx_splits"] = [list_splits[i,1]:list_splits[i,2] for i = 1:size(list_splits,1)]
        dict_["trace_original"] = trace_original
        dict_["trace_array_F20"] = trace_F20
        dict_["trace_array"] = trace
        dict_["n_neuron"] = size(trace, 1)
        dict_["n_t"] = size(trace, 2)

        # behavior
        dict_["velocity"] = velocity
        dict_["θh"] = head_angle
        dict_["pumping"] = pumping
        dict_["ang_vel"] = angular_velocity
        dict_["curve"] = worm_curvature

        dict_["body_angle"] = body_angle
        dict_["body_angle_all"] = body_angle_all
        dict_["body_angle_absolute"] = body_angle_absolute
        
        for var = ["θh", "curve", "ang_vel", "velocity", "pumping"]
            dict_[var*"_s"] = FlavellBase.standardize(dict_[var])
            dict_["s_"*var] = std(dict_[var])
            dict_["u_"*var] = mean(dict_[var])
        end
        
        # tiiming
        dict_["timestamp_nir"] = read(h5f, "timing/timestamp_nir")
        dict_["timestamp_confocal"] = read(h5f, "timing/timestamp_confocal")
        if haskey(h5f["timing"], "stim_begin_confocal")
            dict_["stim_begin_confocal"] = read(h5f, "timing/stim_begin_confocal")
        end
    end
    
    dict_
end