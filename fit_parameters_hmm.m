function [learned_bnet f_prior f_learn f_guess f_slip] = fit_parameters_hmm(bnet, data)

%read in the data, in format rows=student samples, columns=observed answer
% responses = dlmread(data);
responses = data;

% intial values for EM parameter learning
i_prior = 0.3;
i_learn = 0.15;
i_forget = 0;
i_guess = 0.1;
i_slip = 0.1;

% prior
bnet.CPD{1} = tabular_CPD(bnet, bnet.rep_of_eclass(1), 'CPT', [1-i_prior i_prior]);

% learn/forget
bnet.CPD{3} = tabular_CPD(bnet, bnet.rep_of_eclass(3), 'CPT', [1-i_learn i_forget i_learn 1-i_forget]);

% guess/slip
bnet.CPD{2} = tabular_CPD(bnet, bnet.rep_of_eclass(2), 'CPT', [1-i_guess i_slip i_guess 1-i_slip]);

%add observed data to evidence

ncases = size(responses, 1); %number of samples in data set
ss = bnet.nnodes_per_slice; %nodes per slice

obs_node = bnet.observed; %observed node in each slice

cases = cell(1, ncases); %store evidence

for i=1:ncases
	%get the response and affect data for this case
	response = responses(i,:);
	response = response(find(response)); %strip zeros

	T = size(response,2); %number of timeslices for this student
	cases{i} = cell(ss,T);
	cases{i}(obs_node,:) = num2cell(response); %populate with response data
end

% % learn parameters

% initialize dbn parameter learning
engine = smoother_engine(jtree_2TBN_inf_engine(bnet));

% max iterations for EM parameter fitting
max_iter = 10;

[bnet, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);

% values of fit parameters
f_prior = CPD_to_CPT(bnet.CPD{1});
f_prior = f_prior(2);

f_trans = CPD_to_CPT(bnet.CPD{3});
f_learn = f_trans(3);
f_forget = f_trans(2);

f_emit = CPD_to_CPT(bnet.CPD{2});
f_guess = f_emit(3);
f_slip = f_emit(2);

fprintf('intial params:\t prior: %.3f, learn: %.3f, forget: %.3f, guess: %.3f, slip: %.3f\n',...
   i_prior, i_learn, i_forget, i_guess, i_slip);

fprintf('learned params:\t prior: %.3f, learn: %.3f, forget: %.3f, guess: %.3f, slip: %.3f\n',...
   f_prior, f_learn, f_forget, f_guess, f_slip);

learned_bnet = bnet;

