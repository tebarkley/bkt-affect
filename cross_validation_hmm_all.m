function mean_accuracy = cross_validation_hmm_all(skills_list_file, output_file)

% create a file in which to store the parameters and predictions for all of the skills
result_file = output_file;
header = {'skill','num_students','prior','learn','guess','slip','mae','baseline_accuracy','accuracy',...
 'false_positive_rate', 'false_negative_rate'};
fid = fopen(result_file,'w');
fprintf(fid, '%s,',header{1,1:end-1});
fprintf(fid, '%s\n', header{1,end});
fclose(fid);

%read in a list of all of the relevant skills, in order to loop through them, do the cross-validation, and output results to a file
% fid = fopen(skills_list_file,'r');
% skills_list =  textscan(fid, '%s');
% skills_list = skills_list{1};
skills_list = {'box_and_whisker'}

for sk=1:length(skills_list)
% for sk=1:1
	skill = skills_list{sk} %skill name
	
	%create dbn
	bnet = make_knowledge_model_hmm();

	%read in the data for this skill
	response_file = strcat(skill,'_responses.txt')
	responses = dlmread(response_file);

	ncases = size(responses,1); %number of students

	%split data into training and testing sets using crossvalind
	indices = crossvalind('Kfold', ncases, 5); %5 folds, dividing students
	for i=1:5
		test_indices = (indices == i);
		train_indices = ~test_indices;
		train_set = responses(train_indices,:);
		test_set = responses(test_indices,:);

		%learn parameters on the training set and store to take average later
		[learned_bnet f_prior f_learn f_guess f_slip] = fit_parameters_hmm(bnet, train_set);
		prior(i) = f_prior;
		learn(i) = f_learn;
		guess(i) = f_guess;
		slip(i) = f_slip;

		%perform prediction on the test set
		[mae, accuracy, fpr, fnr] = predict_hmm(learned_bnet, test_set);
		maes(i) = mae;
		accuracies(i) = accuracy;
		fprs(i) = fpr;
		fnrs(i) = fnr;
	end

	%print out the results across all folds
	fprintf('\nSkill: %s/n', skill);
	maes
	accuracies
	fprs
	fnrs
	prior
	learn
	guess
	slip

	mean_mae = mean(maes);
	mean_accuracy = mean(accuracies);
	mean_fprs = mean(fprs);
	mean_fnrs = mean(fnrs);
	mean_prior = mean(prior);
	mean_learn = mean(learn);
	mean_guess = mean(guess);
	mean_slip = mean(slip);

	fprintf('\nSkill: %s\n', skill);

	%mean absolute error
	fprintf('\nMean MAE: %.3f\n', mean_mae);

	%Baseline accuracy. How will would we predict if we just predicted the most likely class?
	baseline_accuracy = mean(mean(responses(find(responses))-1)); %this is the percentage of correct questions
	if baseline_accuracy < 0.5
		baseline_accuracy = 1-baseline_accuracy;
	end
	baseline_accuracy=baseline_accuracy*100;
	fprintf('Baseline Accuracy: %.1f\n', baseline_accuracy);

	%accuracy
	fprintf('Mean Accuracy: %.1f\n', mean_accuracy);

	%false positive rate
	fprintf('Mean False Positive Rate: %.1f\n', mean_fprs);

	%false negative rate
	fprintf('Mean False Negative Rate: %.1f\n', mean_fnrs);

	%mean prior
	fprintf('Mean Prior of Knowledge: %.3f\n', mean_prior);

	%mean prior
	fprintf('Mean Learning Transition: %.3f\n', mean_learn);

	%mean guess
	fprintf('Mean Guess: %.3f\n', mean_guess);

	%mean slip
	fprintf('Mean Slip: %.3f\n', mean_slip);

	row_to_write = {skill,ncases,mean_prior,mean_learn,mean_guess,mean_slip,mean_mae,baseline_accuracy,accuracy,mean_fprs,mean_fnrs};
	fid = fopen(result_file,'a');
	fprintf(fid,'%s,',row_to_write{1,1});
	fprintf(fid, '%.3f,', row_to_write{1,2:end-1});
	fprintf(fid, '%.3f\n', row_to_write{1,end});
	fclose(fid);

end



