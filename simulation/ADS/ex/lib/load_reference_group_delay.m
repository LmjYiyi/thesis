function ref = load_reference_group_delay(cfg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 加载参考群时延曲线（预留接口）
% cfg.reference.type = 'none' | 's2p' | 'manual'
% 返回：ref.f_hz, ref.tau_s（均为列向量，或 type='none' 时为空）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ref.f_hz  = [];
ref.tau_s = [];

switch cfg.reference.type
    case 'none'
        return;
    case 's2p'
        % TODO: 读取 .s2p 文件并提取群时延
        warning('load_reference_group_delay: s2p 支持尚未实现');
    case 'manual'
        % TODO: 从 cfg.reference.f_hz / cfg.reference.tau_s 读取手动数据
        if isfield(cfg.reference, 'f_hz') && isfield(cfg.reference, 'tau_s')
            ref.f_hz  = cfg.reference.f_hz(:);
            ref.tau_s = cfg.reference.tau_s(:);
        end
    otherwise
        warning('load_reference_group_delay: 未知 type = %s', cfg.reference.type);
end

end
