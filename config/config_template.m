% 配置文件模板，复制为 config.m 并修改数据路径
function cfg = config()
    % 获取当前文件所在目录的上一级，项目根目录
    root = fileparts(fileparts(mfilename('fullpath')));
    cfg.data_root = fullfile(root, 'data');
    cfg.male_dir = fullfile(cfg.data_root, 'male');
    cfg.female_dir = fullfile(cfg.data_root, 'female');
    cfg.results_dir = fullfile(root, 'results');
    if ~exist(cfg.results_dir, 'dir')
        mkdir(cfg.results_dir);
    end
end
