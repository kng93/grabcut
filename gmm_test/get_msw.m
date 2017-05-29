function [mean1, sigma1, weights1, mean2, sigma2, weights2] = get_msw(fn)
    % Get data from the file
    fid = fopen(fn,'r');
    line = fgetl(fid);
    in_data1 = [];
    in_data2 = [];
    spc = 0;
    
    % Extract data
    while ischar(line)
        val = str2double(line);
        if (~isnan(val))
            in_data2 = [in_data2; val];
        else % Count how many blank spaces
            spc = spc + 1;
        end
        
        % If hit break - set new in_data
        if (spc == 3)
           in_data1 = in_data2;
           in_data2 = [];
        end
        
        line = fgetl(fid);
    end
    fclose(fid);
    
    % Put the data into matrices to return
    num = size(in_data1,1)/3;
    mean1 = in_data1(1:num);
    weights1 = in_data1(num+1:num*2);
    sigma1 = in_data1(num*2+1:end);
      
    mean2 = in_data2(1:num);
    weights2 = in_data2(num+1:num*2);
    sigma2 = in_data2(num*2+1:end);
    
%     
%     disp(['mu=', mat2str(mean1), ', s=', mat2str(sigma1), ', w=', mat2str(weights1)]);
%     disp(['mu=', mat2str(mean2), ', s=', mat2str(sigma2), ', w=', mat2str(weights2)]);

    % Set sigmas
    temp_sig = zeros(1,1, size(sigma1,1));
    temp_sig(1,1,:) = sigma1;
    sigma1 = temp_sig;
    
    temp_sig = zeros(1,1, size(sigma2,1));
    temp_sig(1,1,:) = sigma2;
    sigma2 = temp_sig;
end