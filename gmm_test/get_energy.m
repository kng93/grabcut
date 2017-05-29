function [energy] = get_energy(fg_mdl, bg_mdl, fg_points, bg_points)
    [~, fg_log] = posterior(fg_mdl, fg_points);
    %disp(num2str(fg_log));
    [~, bg_log] = posterior(bg_mdl, bg_points);
    %disp(num2str(bg_log));
    energy = fg_log + bg_log;
end