function mean_accuracy = cross_validation_affect_all(skills_list_file, output_file)

% create a file in which to store the parameters and predictions for all of the skills
result_file = output_file;
header = {'skill','affect','num_students','prior','learn','guess_noaffect','guess_affect','slip_noaffect','slip_affect', ...
'mae','baseline_accuracy','accuracy','false_positive_rate', 'false_negative_rate'};
fid = fopen(result_file,'w');
fprintf(fid, '%s,',header{1,1:end-1});
fprintf(fid, '%s\n', header{1,end});
fclose(fid);

%read in a list of all of the relevant skills, in order to loop through them, do the cross-validation, and output results to a file
fid = fopen(skills_list_file,'r');
skills_list =  textscan(fid, '%s');
skills_list = skills_list{1};
affect_list = {'bored','frustrated','confused','concentrating'};
for sk=1:length(skills_list)
	skill = skills_list{sk} %skill name
	
	%read in the data for this skill
	response_file = strcat(skill,'_responses.txt');
	responses = dlmread(response_file);
	ncases = size(responses,1); %number of students

	%loop through each affect to learn and predict

	for af=1:length(affect_list)
		affect = affect_list{1,af}
		affect_file = strcat(skill,'_',affect,'.txt');
		affects = dlmread(affect_file);

		%create dbn
		bnet = make_knowledge_model_affect();

		%split data into training and testing sets using crossvalind
		indices = crossvalind('Kfold', ncases, 5); %5 folds, dividing students
		for i=1:5
			test_indices = (indices == i);
			train_indices = ~test_indices;
			response_train_set = responses(train_indices,:);
			affect_train_set = affects(train_indices,:);
			response_test_set = responses(test_indices,:);
			affect_test_set = affects(test_indices,:);

			%learn parameters on the training set and store to take average later
			[learned_bnet f_prior f_learn f_guess_noaffect, f_guess_affect, f_slip_noaffect, f_slip_affect, f_affect] ...
			 = fit_parameters_affect(bnet, response_train_set, affect_train_set);
			prior(i) = f_prior;
			learn(i) = f_learn;
			guess_noaffect(i) = f_guess_noaffect;
			guess_affect(i) = f_guess_affect;
			slip_noaffect(i) = f_slip_noaffect;
			slip_affect(i) = f_slip_affect;
			affect_prior(i) = f_affect;

			%perform prediction on the test set
			[mae, accuracy, fpr, fnr] = predict_affect(learned_bnet, response_test_set, affect_test_set);
			maes(i) = mae;
			accuracies(i) = accuracy;
			fprs(i) = fpr;
			fnrs(i) = fnr;
		end

		%print out the results across all folds
		maes
		accuracies
		fprs
		fnrs
		prior
		learn
		guess_noaffect
		guess_affect
		slip_noaffect
		slip_affect
		affect_prior

		mean_accuracy = mean(accuracies);
		mean_mae = mean(maes);
		mean_fprs = mean(fprs);
		mean_fnrs = mean(fnrs);
		mean_prior = mean(prior);
		mean_learn = mean(learn);
		mean_guess_noaffect = mean(guess_noaffect);
		mean_guess_affect = mean(guess_affect);
		mean_slip_noaffect = mean(slip_noaffect);
		mean_slip_affect = mean(slip_affect);
		mean_affect_prior = mean(affect_prior);


		fprintf('\nSkill: %s\n', skill);
		fprintf('\nAffect: %s\n', affect);

		%mean absolute error
		fprintf('\nMean MAE: %.3f\n', mean_mae);

		%Baseline accuracy. How will would we predict if we just predicted the most likely class?
		baseline_accuracy = mean(mean(responses(find(responses))-1)); %this is the percentage of correct questions
		if baseline_accuracy < 0.5
			baseline_accuracy = 1-baseline_accuracy
		end
		baseline_accuracy = baseline_accuracy*100;
		fprintf('Baseline Accuracy: %.1f\n', baseline_accuracy);

		%accuracy
		fprintf('Mean Accuracy: %.1f\n', mean_accuracy);

		%false positive rate
		fprintf('Mean False Positive Rate: %.1f\n', mean_fprs);

		%false negative rate
		fprintf('Mean False Negative Rate: %.1f\n', mean_fnrs);

		%mean prior
		fprintf('Mean Prior of Knowledge: %.3f\n', mean_prior);

		%mean learn
		fprintf('Mean Learning Transition: %.3f\n', mean_learn);

		%mean guess, affect not active
		fprintf('Mean Guess No Affect: %.3f\n', mean_guess_noaffect)

		%mean guess, affect active
		fprintf('Mean Guess With Affect: %.3f\n', mean_guess_affect);

		%mean slip, affect not active
		fprintf('Mean Slip No Affect: %.3f\n', mean_slip_noaffect);

		%mean slip, affect active
		fprintf('Mean Slip Affect: %.3f\n', mean_slip_affect);

		%prior of affect
		fprintf('Prior Probability of Affect: %.3f\n', mean_affect_prior);
		row_to_write = {skill,affect,ncases,mean_prior,mean_learn,mean_affect_prior,mean_guess_noaffect,mean_guess_affect,mean_slip_noaffect,mean_slip_affect,...
		mean_mae,baseline_accuracy,accuracy,mean_fprs,mean_fnrs};
		fid = fopen(result_file,'a');
		fprintf(fid,'%s,',row_to_write{1,1:2});
		fprintf(fid, '%.3f,', row_to_write{1,3:end-1});
		fprintf(fid, '%.3f\n', row_to_write{1,end});
		fclose(fid);
	end
end

