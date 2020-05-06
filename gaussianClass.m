function [positive_likelihood, negative_likelihood, training_pos_prior, training_neg_prior, test_error, mean_trainingPos, cov_trainingPos, mean_trainingNeg, cov_trainingNeg] = gaussianClass(cov_matrix_type, x, discriminant_type)
    
    % Shuffle the dataset and split into training and testing
    [m,n] = size(x);
    P = 0.50;
    index = randperm(m);
    trainingDataset = x(index(1:round(P*m)),:);
    testingDataset = x(index(round(P*m)+1:end),:);
    
    % Split training dataset into positive and negative for diabetes
    trainingDatasetPos = trainingDataset(trainingDataset(:,9) == 1,:);
    trainingDatasetNeg = trainingDataset(trainingDataset(:,9) == 0,:);
    
    % Calculate the mean vector for each class
    trainingPosMean = mean(trainingDatasetPos(:, 1:8));
    trainingNegMean = mean(trainingDatasetNeg(:, 1:8));
    
    % Calculate the covariance matrix for each class
    trainingPosCov = cov(trainingDatasetPos(:, 1:8));
    trainingNegCov = cov(trainingDatasetNeg(:, 1:8));
    
    % Calculate Priors based on dataset size
    trainingPosPrior = length(trainingDatasetPos)/length(trainingDataset);
    trainingNegPrior = length(trainingDatasetNeg)/length(trainingDataset);
    
    % Calculate the multivariate Gaussian densities
    switch discriminant_type
        case "quadratic"
            %Using the mean vector and covariance matrix from the training
            %dataset, evaluate the Gaussian density at each row of the
            %training dataset.
            testingDatasetPosLikelihood = mvnpdf(testingDataset(:, 1:8), trainingPosMean, trainingPosCov);
            testingDatasetNegLikelihood = mvnpdf(testingDataset(:, 1:8), trainingNegMean, trainingNegCov);
        case "linear"
            switch cov_matrix_type
                case "full"
                    
                case "diagonal"
                    
                case "diagonal equal"
                    
            end
    end
    
    % Calculate the priors
    loss_tracking = []; % store 0 if correct, store 1 if incorrect
    for i=1:length(testingDataset)
        test_label = testingDataset(i, 9);
        posterior_positive_diabetes = (testingDatasetPosLikelihood * trainingPosPrior)/((testingDatasetPosLikelihood * trainingPosPrior)+(testingDatasetNegLikelihood * trainingNegPrior));
        
        if posterior_positive_diabetes > 0.5
            classification = 1;
        else 
            classification = 0;
        end
        
        loss_tracking = [loss_tracking, abs(classification - test_label)];
          
    end
    test_error = sum(loss_tracking)/length(loss_tracking);
    
    positive_likelihood = testingDatasetPosLikelihood;
    negative_likelihood = testingDatasetNegLikelihood;
    
    training_pos_prior = trainingPosPrior;
    training_neg_prior = trainingNegPrior;
    
    mean_trainingPos = trainingPosMean;
    cov_trainingPos = trainingPosCov;
    mean_trainingNeg = trainingNegMean;
    cov_trainingNeg = trainingNegCov;
    
    
end

