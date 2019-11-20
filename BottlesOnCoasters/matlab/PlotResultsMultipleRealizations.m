function PlotResultsMultipleRealizations()

    %% Parameters
    
    resultDir = '../results-clutter';
    metricNames = {'nGraspedObjects', 'nPlacedObjects'};
    resultLegend = {'Grasped', 'Placed'};
    episodeBlock = 1000; worstPlotValue = 0; bestPlotValue = 2;

    %% Load

    close('all');
    
    resultFiles = dir([resultDir '/*.mat']);
    data = cell(1, length(resultFiles));
    for idx=1:length(resultFiles)
        name = resultFiles(idx).name(1:end-4);
        fullName = [resultFiles(idx).folder '/' name];
        data{idx} = load(fullName);
    end

    %% Plot Task Success
    
    figure;  hold('on');
    plots = zeros(1, length(metricNames));
    colors = get(gca, 'ColorOrder');
    maxMinRealizationLength = -inf;
    
    for idx=1:length(metricNames)
        
        nRealizations = length(data);
        
        % determine length of shortest learning curve in this scenario
        minRealizationLength = inf;
        for jdx=1:nRealizations
            metric = eval(['data{jdx}.' metricNames{idx}]);
            if length(metric) < minRealizationLength
                minRealizationLength = length(metric);
            end
        end
        
        % determine longest plot in comparison
        if minRealizationLength > maxMinRealizationLength
            maxMinRealizationLength = minRealizationLength;
        end
        
        metricAll = zeros(nRealizations, minRealizationLength);
    
        for jdx=1:nRealizations
            metric = eval(['data{jdx}.' metricNames{idx}]);
            metricAll(jdx, 1:minRealizationLength) = ...
                metric(1:minRealizationLength);
        end

        uP = mean(metricAll, 1);
        sP = std(metricAll, 0, 1);

        uRvis = reshape(uP, [episodeBlock, length(uP)/episodeBlock]);
        uRvis = mean(uRvis, 1);
        sRvis = reshape(sP, [episodeBlock, length(sP)/episodeBlock]);
        sRvis = mean(sRvis, 1);

        x = (0:episodeBlock:length(uP)-1);

        fill([x'; flipud(x')],[(uRvis-sRvis)'; flipud((uRvis+sRvis)')], ...
            colors(idx, :), 'linestyle','none'); alpha(0.30);
        plots(idx) = plot(x, uRvis, 'color', colors(idx, :),'linewidth', 2);
    end
    
    xlim([0, maxMinRealizationLength-episodeBlock]);
    ylim([worstPlotValue, bestPlotValue]); grid('on');
    xlabel('Episode', 'FontWeight', 'bold');
    ylabel('Number of Objects', 'FontWeight', 'bold');
    
    if ~isempty(resultLegend)
        legend(plots, resultLegend);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Print Quantitative Results
    
    disp('--------------------------------------------------------------');
    for idx=1:length(metricNames)
        nRealizations = length(data);
        nEpisodes = 0; totalMetric = 0;
        for jdx=1:nRealizations
            metric = eval(['data{jdx}.' metricNames{idx}]);
            totalMetric = totalMetric + sum(metric( ...
                data{jdx}.unbiasOnEpisode+1:end));
            nEpisodes = nEpisodes + length(metric( ...
                data{jdx}.unbiasOnEpisode+1:end));
        end
        disp([metricNames{idx} ': ' num2str(totalMetric/nEpisodes) '.']);
    end
    