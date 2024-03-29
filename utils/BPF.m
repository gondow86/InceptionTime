%% Doppler radar signal ----------------------------------------------------
% ID1
%Doppler = dlmread('../data/Keio Hospital/Doppler/ID03/Doppler010421');
Doppler = dlmread("C:\Users\grpro\workspace\grad_thesis\InceptionTime\data\Keio Hospital\Doppler\ID12\20220703①_20220703_113740.csv");
ex_name = 'ID1';

R_Iraw = double(Doppler(:,1))/2048;                                                              % In-phase data of Doppler radar signal
R_Qraw = double(Doppler(:,2))/2048;                                                              % Quadrature data of Doppler radar signal
fs = 1000;

R_I = R_Iraw;                                                              % 観測時間のドップラーレーダ同位相信号
R_Q = R_Qraw;                                                              % 観測時間のドップラーレーダ直交信号
R = R_I - R_Q*1j;                                                          % 観測時間のドップラーレーダ複素数信号
% phase = unwrap(atan(R_Iraw./R_Qraw)*2)/2; % Phase signal
len_R = length(R);                                                         % 観測時間のドップラーレーダサンプリング数

T = length(R_Iraw); % Data length
obtime = T/fs; % Observation duration
Tob = 1/fs:1/fs:obtime; % Time data vector for figure

sub = 25;
Fcut_low1  = 0.3;
Fcut_high1 = 100;
bpFilt_v = designfilt('bandpassfir','FilterOrder',floor(100000/sub),...
    'CutoffFrequency1',Fcut_low1,'CutoffFrequency2',Fcut_high1,...
    'SampleRate',fs);
Delay_v = mean(grpdelay(bpFilt_v));
filtered = filter(bpFilt_v,[R; zeros(Delay_v,1)]);
R_bpf = filtered(Delay_v+1:end);
R_bpf_abs = abs(R_bpf);
R_abs = abs(R);
%R_bpf_abs_norm = normalize(R_bpf_abs, "range");
%findpeaks(R_I, "MinPeakHeight", 0.9);
[pks, locs] = findpeaks(R_I, "MinPeakHeight", 0.9);

%writematrix(locs, "I_raw_01_peaks.csv");
%writematrix(real(R), "I_raw_12.csv");
writematrix(real(R), "Q_raw_12.csv");
%writematrix(R_bpf_abs, "filtered_abs.csv");
%writematrix(R_abs, "nonfilterd_abs.csv");