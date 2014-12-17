function mean_accuracy = cross_validation_hmm(response_file)

%create dbn
bnet = make_knowledge_model_hmm();

%read in the data, in format rows=student samples, columns=observed answer
if response_file(end) == 'v'
	responses = dlmread(response_file, ','); %read in csv
else
	responses = dlmread(response_file); %read in txt
end
responses = responses(1:1000,:);

% %need to get rid of responses that only have one question answer, as this is breaking the unrolled_dbn
% for row=1:size(responses,1)
% 	response = responses(row,:);
% 	response = response(find(response));
% 	if length(response) <= 1
% 		keep(row) = 0;
% 	else
% 		keep(row) = 1;
% 	end
% end

% responses = responses(keep==1,:);

ncases = size(responses,1);

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
maes
accuracies
fprs
fnrs
prior
learn
guess
slip

mean_accuracy = mean(accuracies);

%mean absolute error
fprintf('\nMean MAE: %.3f\n', mean(maes));

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
fprintf('Mean False Positive Rate: %.1f\n', mean(fprs));

%false negative rate
fprintf('Mean False Negative Rate: %.1f\n', mean(fnrs));

%mean prior
fprintf('Mean Prior of Knowledge: %.3f\n', mean(prior));

%mean prior
fprintf('Mean Learning Transition: %.3f\n', mean(learn));

%mean guess
fprintf('Mean Guess: %.3f\n', mean(guess));

%mean slip
fprintf('Mean Slip: %.3f\n', mean(slip));


