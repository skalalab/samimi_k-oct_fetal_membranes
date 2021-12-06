clc;
%close all;
% clear variables;

[fname, path] = uigetfile('*.mat', 'Load tiff.');
load(strcat(path,fname),'-mat');

ind_toe_low = find(Pressure>=0.5, 1);                     % indices for the toe region
ind_toe_high = find(Pressure>=5, 1);
if isempty(ind_toe_high)
    ind_toe_high = find(Pressure==max(Pressure(:)), 1);
end

ind_loaded_low = find(Pressure>=7.5, 1);
ind_loaded_high = find(Pressure==max(Pressure(:)), 1);    % indices for the under-load region
%ind_loaded_high = find(Pressure>=17.8, 1);    % indices for the under-load region

C_toe = cat(2, Strain(ind_toe_low:ind_toe_high), ones(ind_toe_high-ind_toe_low+1,1));
d_toe = Stress(ind_toe_low:ind_toe_high)*InitialThickness;
lin_coeffs_toe = C_toe\d_toe;                       % solve for linear fit coefficients
TensionMod_toe = lin_coeffs_toe(1);                 % tangent modulus of the toe region in [N/m]


C_loaded = cat(2, Strain(ind_loaded_low:ind_loaded_high), ones(ind_loaded_high-ind_loaded_low+1,1));
d_loaded = Stress(ind_loaded_low:ind_loaded_high)*InitialThickness;
lin_coeffs_loaded = C_loaded\d_loaded;              % solve for linear fit coefficients
TensionMod_loaded = lin_coeffs_loaded(1);           % tangent modulus of the under-load region in [N/m]

%%  Plot results (fix tension units)

scrsz = get(0,'ScreenSize');
fig = figure('Position',[round(0.1*scrsz(3)) round(0.1*scrsz(4)) round(0.8*scrsz(3)) round(0.8*scrsz(4))]);     % Monitor 1
%fig = figure('Position',[round(1.1*scrsz(3)) round(0.1*scrsz(4)) round(0.8*scrsz(3)) round(0.8*scrsz(4))]);    % Monitor 2
plot(Strain, Stress*InitialThickness, 'LineWidth', 2, 'Color', 'b'); hold on; ax=gca;ax.FontSize=25;ax.LineWidth=2; ylabel('Tension [N/m]', 'Interpreter', 'latex', 'FontSize', 25); xlabel('Strain', 'Interpreter', 'latex', 'FontSize', 25); hold on;
title(fname(1:end-22), 'Interpreter', 'none', 'FontSize', 16);
plot(C_toe(:,1), d_toe, C_toe(:,1), C_toe*lin_coeffs_toe, 'LineWidth', 2, 'Color', [255,165,0]/255);
plot(C_loaded(:,1), d_loaded, C_loaded(:,1), C_loaded*lin_coeffs_loaded, 'LineWidth', 2, 'Color', [255,0,0]/255);

str = char(num2bank(round(TensionMod_toe))); txt1 = sprintf('Tension Modulus\n= %s [N/m]',str(1:end-3));
text(0.6*Strain(ind_toe_low)+0.4*Strain(ind_toe_high),(0.85*Stress(ind_toe_high)+0.15*Stress(ind_toe_low))*InitialThickness,txt1,'HorizontalAlignment','center', 'Color', [255,165,0]/255, 'Interpreter', 'latex', 'FontSize', 25)

str = char(num2bank(round(TensionMod_loaded))); txt2 = sprintf('Tension Modulus\n= %s [N/m]',str(1:end-3));
text(0.75*Strain(ind_loaded_low)+0.25*Strain(ind_loaded_high),(0.85*Stress(ind_loaded_high)+0.15*Stress(ind_loaded_low))*InitialThickness,txt2,'HorizontalAlignment','center', 'Color', [255,0,0]/255, 'Interpreter', 'latex', 'FontSize', 25)

%%
display(round(TensionMod_loaded))
display(round(TangMod_loaded*InitialThickness))
warning('on');
if (round(TensionMod_loaded)~=round(TangMod_loaded*InitialThickness))
    warning('Pressure range should be adjusted!')
end
clipboard('copy',round(TensionMod_loaded))
