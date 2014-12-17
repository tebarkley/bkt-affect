function mean_accuracy = cross_validation_affect(response_file, affect_file)

%create dbn
bnet = make_knowledge_model_affect();

%read in the data, in format rows=student samples, columns=observed answer
if response_file(end) == 'v'
	responses = dlmread(response_file, ','); %read in csv
else
	responses = dlmread(response_file); %read in txt
end
if affect_file(end) == 'v'
	affects = dlmread(affect_file, ','); %read in csv
else
	affects = dlmread(affect_file); %read in txt
end
responses = responses(1:1000,:);
affects = affects(1:1000,:);
ncases = size(responses,1);

% for n=1:ncases
% 	response = responses(n,:);
% 	response = response(find(response));
% 	affect = affects(n,:);
% 	affect = affect(find(affect));
% 	T_r = size(response,2);
% 	T_a = size(affect,2);
% 	if T_r~=T_a
% 		T_r
% 		T_a
% 		n
% 	end
% end

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

	%perform prediction on the test set. Either adds affect in after the node had been predicted, or during prediction

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

%mean absolute error
fprintf('\nMean MAE: %.3f\n', mean(maes));

%Baseline accuracy. How will would we predict if we just predicted the most likely class?
baseline_accuracy = mean(mean(responses(find(responses))-1)); %this is the percentage of correct questions
if baseline_accuracy < 0.5
	baseline_accuracy = 1-baseline_accuracy
end
baseline_accuracy = baseline_accuracy*100
fprintf('Baseline Accuracy: %.1f\n', baseline_accuracy);

%accuracy
fprintf('Mean Accuracy: %.1f\n', mean_accuracy);

%false positive rate
fprintf('Mean False Positive Rate: %.1f\n', mean(fprs));

%false negative rate
fprintf('Mean False Negative Rate: %.1f\n', mean(fnrs));

%mean prior
fprintf('Mean Prior of Knowledge: %.3f\n', mean(prior));

%mean learn
fprintf('Mean Learning Transition: %.3f\n', mean(learn));

%mean guess, affect not active
fprintf('Mean Guess No Affect: %.3f\n', mean(guess_noaffect))

%mean guess, affect active
fprintf('Mean Guess With Affect: %.3f\n', mean(guess_affect));

%mean slip, affect not active
fprintf('Mean Slip No Affect: %.3f\n', mean(slip_noaffect));

%mean slip, affect active
fprintf('Mean Slip Affect: %.3f\n', mean(slip_affect));

%prior of affect
fprintf('Prior Probability of Affect: %.3f\n', mean(affect_prior));

