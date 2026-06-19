function cfg = config()
    % 自动获取当前文件所在目录，即项目根目录
    root = fileparts(mfilename('fullpath'));
    cfg.data_root = fullfile(root, 'data');
    cfg.male_dir = fullfile(cfg.data_root, 'male');
    cfg.female_dir = fullfile(cfg.data_root, 'female');
    cfg.results_dir = fullfile(root, 'results');
    if ~exist(cfg.results_dir, 'dir')
        mkdir(cfg.results_dir);
    end
end
