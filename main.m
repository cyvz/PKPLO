%% CEC17 Benchmark Testing with Status-Based Optimization (SBO)

addpath(genpath(pwd));

% Parameters: 
POP_SIZE = 30; MAX_FES = POP_SIZE * 10000; DIM = 30; LB = -100; UB = 100;

% Initialize results storage
results = struct('fnum',[],'best_pos',[],'best_val',[],'runtime',[]);
start_time = tic;

func_num = 1;
fobj = @(x) cec17_func(x', func_num);
func_start = tic;
    
% Run optimization
[best_pos, convergence_curve] = PKPLO(POP_SIZE, MAX_FES, LB, UB, DIM, fobj);
    
% Store results
results(func_num).fnum = func_num;
results(func_num).best_pos = best_pos;

% Display function results
fprintf('Best Fitness: %.3e\n', convergence_curve(end));
fprintf('Best Solution: [%s]\n', strjoin(arrayfun(@(x) sprintf('%.3f',x), best_pos, 'UniformOutput', false), ', '));