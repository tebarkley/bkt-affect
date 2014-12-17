function [mae accuracy fpr fnr] = predict_affect(bnet, response_data, affect_data)

%read in the data, in format rows=student samples, columns=observed answer
% responses = dlmread(response_file);
responses = response_data;
affects = affect_data;

ncases = size(responses, 1); %number of student samples in the data
ss = bnet.nnodes_per_slice; %get the number of nodes in a slice
obs_node = bnet.observed; %observed node
total_responses = 0; %variable to count the number of questions*students

%loop through the students to perform inference and get the knowledge and question response predictions
for s=1:ncases %loops through students
	%get student data
	actual_responses = responses(s,:);
	actual_responses = actual_responses(find(actual_responses)); %get responses, removing padded zeros
	actual_affects = affects(s,:);
	actual_affects = actual_affects(find(actual_affects)); %get affects, removing padded zeros
	T = size(actual_responses, 2); %number of timeslices for that student 
	if T==1
		unrolled_size = 2; %unroll DBN to two time slices when there is only one observed response
	else
		unrolled_size = T;
	end
	total_responses = total_responses + T; %increment the total number of responses

	%define the engine
	engine = jtree_unrolled_dbn_inf_engine(bnet, unrolled_size);

	%initialize evidence as empty array
	%in doing this, we are assuming that some of the affect is measured in the previous question
	evidence = cell(ss,unrolled_size);

	%initialize arrays in which to store various prediction results
	s_pred_performance_perc=[];
	s_pred_knowledge_perc=[];
	s_pred_performance=[];
	s_pred_knowledge=[];

	%loop through t to T, performing inference and adding observed data to evidence
	for t=1:T
		[engine, ll] = enter_evidence(engine, evidence); %perform inference
		P = marginal_nodes(engine,2,t); %get the probabilities for the knowledge node at the t timeslice
		p_k = P.T(2); %probability of knowledge
		k_predict = find(P.T==max(P.T)); %output the most likely knowledge state: 1 or 2
		P = marginal_nodes(engine,3,t); %get the probabilities of question node at the t timeslice
		p_q = P.T(2); %probability of correct
		q_predict = find(P.T==max(P.T));

		%add the inferred values to the appropriate student results cells
		s_pred_performance_perc(t) = p_q;
		s_pred_knowledge_perc(t) = p_k;
		s_pred_performance(t) = q_predict;
		s_pred_knowledge(t) = k_predict;

		%add student's actual responses and affects to the evidence
		evidence{3, t} = actual_responses(t);
		evidence{1, t} = actual_affects(t);
	end 

	%calculate error metrics for each student, which will be aggregated at the end
	absolute_errors(s) = sum(abs((actual_responses-1) - s_pred_performance_perc));
	total_correct(s) = sum(actual_responses == s_pred_performance);
	false_positives(s) = sum((s_pred_performance-actual_responses)==1);
	false_negatives(s) = sum((actual_responses-s_pred_performance)==1);

end

%calculate and output the error metric
mae = sum(absolute_errors)/total_responses;
accuracy = (sum(total_correct)/total_responses)*100;
fpr = (sum(false_positives)/total_responses)*100;
fnr = (sum(false_negatives)/total_responses)*100;

% %mean absolute error
% fprintf('\nMAE: %.3f\n\n', mae);

% %accuracy
% fprintf('Percent Responses Estimated Correctly: %.1f\n\n', accuracy);

% %false positive rate
% fprintf('False Positive Rate: %.1f\n\n', fpr);

% %false negative rate
% fprintf('False Negative Rate: %.1f\n\n', fnr);