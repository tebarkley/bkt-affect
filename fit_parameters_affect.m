function [learned_bnet f_prior f_learn f_guess_noaffect, f_guess_affect, f_slip_noaffect, f_slip_affect, f_affect]= fit_parameters_affect(bnet, response_data, affect_data)

responses = response_data;
affects = affect_data;

% intial values for EM parameter learning
i_prior = 0.3;
i_learn = 0.15;
i_forget = 0;

i_guess_affect = 0.1;
i_guess_noaffect = 0.1;
i_slip_noaffect = 0.1;
i_slip_affect = 0.1;

i_affect = mean(mean(affects(find(affects))-1)); %the prior of affect is the percentage of observed data with that affect

%affect probabilities
bnet.CPD{1} = tabular_CPD(bnet, bnet.rep_of_eclass(1), 'CPT', [1-i_affect, i_affect], 'adjustable', 0); %affect prior is fixed at average affect in dataset

% prior of knowledge
bnet.CPD{2} = tabular_CPD(bnet, bnet.rep_of_eclass(2), 'CPT', [1-i_prior i_prior]);

% question emission probabilities
bnet.CPD{3} = tabular_CPD(bnet, bnet.rep_of_eclass(3), 'CPT', [1-i_guess_noaffect, 1-i_guess_affect, i_slip_noaffect, i_slip_affect ...
 i_guess_noaffect, i_guess_affect, 1-i_slip_noaffect, 1-i_slip_affect]);

% learn/forget knowledge transition probabilities
bnet.CPD{4} = tabular_CPD(bnet, bnet.rep_of_eclass(4), 'CPT', [1-i_learn i_forget i_learn 1-i_forget]);

% add observed data to evidence
ncases = size(responses, 1); %number of samples in data set
ss = bnet.nnodes_per_slice; %nodes per slice

obs_node = bnet.observed; %observed node in each slice

cases = cell(1, ncases); %store evidence

for i=1:ncases
	%get the response and affect data for this case
	response = responses(i,:);
	response = response(find(response)); %strip zeros
	% response
	affect = affects(i,:);
	affect = affect(find(affect)); %strip zeros
	% affect

	T = size(response,2); %number of timeslices for this student
	% T = size(affect,2)
	cases{i} = cell(ss,T);
	cases{i}(obs_node(1),:) = num2cell(affect); %populate with affect data
	cases{i}(obs_node(2),:) = num2cell(response); %populate with response data
end

% % learn parameters
% initialize dbn parameter learning
engine = smoother_engine(jtree_2TBN_inf_engine(bnet));

% max iterations for EM parameter fitting
max_iter = 10;

%learn parameters
[bnet, LLtrace] = learn_params_dbn_em(engine, cases, 'max_iter', max_iter);

% values of fit parameters
f_prior = CPD_to_CPT(bnet.CPD{2});
f_prior = f_prior(2); %estimated prior

f_trans = CPD_to_CPT(bnet.CPD{4});
f_learn = f_trans(3); %estimated learn
f_forget = f_trans(2); %estimated forget

f_emit = CPD_to_CPT(bnet.CPD{3});
f_slip_noaffect = f_emit(3);
f_slip_affect = f_emit(4);
f_guess_noaffect = f_emit(5);
f_guess_affect = f_emit(6);

f_affect = CPD_to_CPT(bnet.CPD{1});
f_affect = f_affect(2);

fprintf('intial params:\t prior: %.3f, learn: %.3f, forget: %.3f, guess_noaffect: %.3f, guess_affect: %.3f, slip_no_affect: %.3f, slip_affect: %.3f, affect_prior: %.3f\n',...
   i_prior, i_learn, i_forget, i_guess_noaffect, i_guess_affect, i_slip_noaffect, i_slip_affect, i_affect);

fprintf('learned params:\t prior: %.3f, learn: %.3f, forget: %.3f, guess_noaffect: %.3f, guess_affect: %.3f, slip_no_affect: %.3f, slip_affect: %.3f, affect_prior: %.3f\n',...
   f_prior, f_learn, f_forget, f_guess_noaffect, f_guess_affect, f_slip_noaffect, f_slip_affect, f_affect);

learned_bnet = bnet; %emit the learned
