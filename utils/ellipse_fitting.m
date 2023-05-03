IDrange = 1:12;
for indx = 1:length(IDrange)
    ID_i = sprintf('I_raw_%02d',IDrange(indx));
    ID_q = sprintf('Q_raw_%02d',IDrange(indx));
    fprintf('----------- Loading %s ------------\n', ID);
    
    path_i = append('../data/Keio Hospital/', ID_i, '.csv');
    path_q = append('../data/Keio Hospital/', ID_q, '.csv');
    % search file
    
    radar_i = readmatrix(path_i);
    radar_q = readmatrix(path_q);
    %% Prepare data
     % output.(ID).(scenario) = struct;
     [~,~,~,radar_dist] = elreko(radar_i,radar_q, 0); % Ellipse fitting, compensation and distance reconstruction
end