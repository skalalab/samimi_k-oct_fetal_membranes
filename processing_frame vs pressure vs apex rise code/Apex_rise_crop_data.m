    [fname, path] = uigetfile('*.txt', 'Load PRESSURE text file.');
P = importdata(cat(2,path,fname),';',0);
press = P.data; 
% pressure_interp = transpose(interp1 (1:size(press), press, linspace(1, size(press,1), size(apex_rise_metric,1))));

figure, plot(apex_rise_metric)
[x,y] = ginput;
%x = round(x);

figure, plot(press)
[xx,yy] = ginput;


x_ind = [1;round(x)];
%apex_rise_metric_cropped = apex_rise_metric (1:x(1));

xx_ind = [1;round(xx)];

apex_rise_metric_cropped = [0];
apex_rise_metric_not_cropped =[];
pressure_interp_cropped = [0];
pressure_interp_not_cropped = [];
for i=2:size(x_ind,1)
    %if even index, add to output array
    if ~mod(i,2)
        try
            apex_rise_metric_not_cropped = [apex_rise_metric_not_cropped; apex_rise_metric_cropped(end)-apex_rise_metric(x_ind(i-1))+(apex_rise_metric(x_ind(i-1):x_ind(i))); NaN(x_ind(i+1)-x_ind(i)-1, 1)];
        catch
            apex_rise_metric_not_cropped = [apex_rise_metric_not_cropped; apex_rise_metric_cropped(end)-apex_rise_metric(x_ind(i-1))+(apex_rise_metric(x_ind(i-1):x_ind(i))); NaN(size(apex_rise_metric,1)-x_ind(i), 1)];
        end
        apex_rise_metric_cropped = [apex_rise_metric_cropped; apex_rise_metric_cropped(end)-apex_rise_metric(x_ind(i-1))+(apex_rise_metric(x_ind(i-1):x_ind(i)))];
        
        press_orig = press(xx_ind(i-1):xx_ind(i));
        pressure_interp = transpose(interp1 (1:size(press_orig), press_orig, linspace(1, size(press_orig,1), x_ind(i)-x_ind(i-1)+1)));
        try
            pressure_interp_not_cropped = [pressure_interp_not_cropped; pressure_interp; NaN(x_ind(i+1)-x_ind(i)-1, 1)];
        catch
            pressure_interp_not_cropped = [pressure_interp_not_cropped; pressure_interp; NaN(size(apex_rise_metric,1)-x_ind(i), 1)];
        end
        pressure_interp_cropped = [pressure_interp_cropped; pressure_interp];
    end
    %if odd index, skip
end

scrsz = get(0,'ScreenSize');
figure, plot(apex_rise_metric_cropped)
figure, plot(pressure_interp_cropped)
fig = figure('Position',[round(0.1*scrsz(3)) round(0.1*scrsz(4)) round(0.8*scrsz(3)) round(0.8*scrsz(4))]);     % Monitor 1
plot(apex_rise_metric_cropped,pressure_interp_cropped,'LineWidth', 2, 'Color', 'b'); hold on; ax=gca;ax.FontSize=25;ax.LineWidth=2; ylabel('Pressure [kPa]', 'Interpreter', 'latex', 'FontSize', 25); xlabel('Apex rise [mm]', 'Interpreter', 'latex', 'FontSize', 25); hold on;

%% Save [Apex Rise, Pressure] pairs to CSV file

filename = strcat(fname(1:end-4), '_Apex.csv');
writematrix([apex_rise_metric_not_cropped, pressure_interp_not_cropped], cat(2,path,filename));

