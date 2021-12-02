%%  Stress-Strain Analysis

PoissonRatio = 0.48;
InitialApex = 6;   % in [mm]
InitialThickness = 0.05;   % in [mm]
InitialRadius = (InitialApex^2 + 15^2)/(2*InitialApex);   % in [mm]

Apex = InitialApex + apex_rise_metric_cropped;
Apex = cat(1, InitialApex, Apex);
Pressure = cat(1, 0, pressure_interp_cropped);  % in [kPa]

Radius = (Apex.^2 + 15^2)./(2*Apex);   % in [mm]  *****
Thickness = (InitialThickness*InitialApex*InitialRadius)./(Apex.*Radius);  % in [mm]

Tension = (Pressure.*Radius)/2000;   % in [N/mm]

YoungMod = (1-PoissonRatio)*(Pressure.*Radius.^2)./(2*Apex.*Thickness);   % in [kPa]
Strain = Apex./Radius;% *****
Stress = YoungMod.*Strain;   % in [kPa] *****

%figure, plot(Apex, Pressure);
%figure, plot(Strain, Stress);

ind_toe_low = find(Pressure>=0.5, 1);                     % indices for the toe region
ind_toe_high = find(Pressure>=5, 1);
if isempty(ind_toe_high)
    ind_toe_high = find(Pressure==max(Pressure(:)), 1);
end

ind_loaded_low = find(Pressure>=7.5, 1);
ind_loaded_high = find(Pressure==max(Pressure(:)), 1);    % indices for the under-load region
%ind_loaded_high = find(Pressure>=8.5, 1);    % indices for the under-load region

C_toe = cat(2, Strain(ind_toe_low:ind_toe_high), ones(ind_toe_high-ind_toe_low+1,1));
d_toe = Stress(ind_toe_low:ind_toe_high);
lin_coeffs_toe = C_toe\d_toe;                       % solve for linear fit coefficients
TangMod_toe = lin_coeffs_toe(1);                    % tangent modulus of the toe region in [kPa]


C_loaded = cat(2, Strain(ind_loaded_low:ind_loaded_high), ones(ind_loaded_high-ind_loaded_low+1,1));
d_loaded = Stress(ind_loaded_low:ind_loaded_high);
lin_coeffs_loaded = C_loaded\d_loaded;              % solve for linear fit coefficients
TangMod_loaded = lin_coeffs_loaded(1);              % tangent modulus of the under-load region in [kPa]

%%  Plot results

scrsz = get(0,'ScreenSize');
fig = figure('Position',[round(0.1*scrsz(3)) round(0.1*scrsz(4)) round(0.8*scrsz(3)) round(0.8*scrsz(4))]);     % Monitor 1
%fig = figure('Position',[round(1.1*scrsz(3)) round(0.1*scrsz(4)) round(0.8*scrsz(3)) round(0.8*scrsz(4))]);    % Monitor 2
plot(Strain, Stress, 'LineWidth', 2, 'Color', 'b'); hold on; ax=gca;ax.FontSize=25;ax.LineWidth=2; ylabel('Stress [kPa]', 'Interpreter', 'latex', 'FontSize', 25); xlabel('Strain', 'Interpreter', 'latex', 'FontSize', 25); hold on;
title(fname(1:end-22), 'Interpreter', 'none', 'FontSize', 16);
plot(C_toe(:,1), d_toe, C_toe(:,1), C_toe*lin_coeffs_toe, 'LineWidth', 2, 'Color', [255,165,0]/255);
plot(C_loaded(:,1), d_loaded, C_loaded(:,1), C_loaded*lin_coeffs_loaded, 'LineWidth', 2, 'Color', [255,0,0]/255);

%txt1 = sprintf('Tangent Modulus\n= %.0f [kPa]',TangMod_toe);
str = char(num2bank(round(TangMod_toe))); txt1 = sprintf('Tangent Modulus\n= %s [kPa]',str(1:end-3));
text(0.6*Strain(ind_toe_low)+0.4*Strain(ind_toe_high),0.85*Stress(ind_toe_high)+0.15*Stress(ind_toe_low),txt1,'HorizontalAlignment','center', 'Color', [255,165,0]/255, 'Interpreter', 'latex', 'FontSize', 25)

str = char(num2bank(round(TangMod_loaded))); txt2 = sprintf('Tangent Modulus\n= %s [kPa]',str(1:end-3));
text(0.75*Strain(ind_loaded_low)+0.25*Strain(ind_loaded_high),0.85*Stress(ind_loaded_high)+0.15*Stress(ind_loaded_low),txt2,'HorizontalAlignment','center', 'Color', [255,0,0]/255, 'Interpreter', 'latex', 'FontSize', 25)

%%  Save results to disk

filename = strcat(fname(1:end-4), '_Analysis.mat');
save(filename, 'InitialApex', 'InitialRadius', 'InitialThickness', 'Pressure', 'Apex', 'Radius', 'Thickness', 'Tension', 'YoungMod', 'Strain', 'Stress', 'TangMod_toe', 'TangMod_loaded');
saveas(fig, strcat(fname(1:end-4), '_Stress_Strain.tif'));
