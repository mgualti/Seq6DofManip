function PlotResultsMultipleRealizations()

    %% Parameters
    
    resultDirs = {'../results-mc50', '../results-sarsa50', '../results-qlearning50'};
    resultLegend = {'Monte Carlo', 'Sarsa', 'Q-learning'};
    episodeBlock = 1000; worstPlotValue = 0; bestPlotValue = 2;
    
    saveFilePrefix = '2019-02-04';
    saveFilePostfixes = {'PlacedObjects'};
    figsToSave = [];

    %% Load

    close('all');
    figs = [];
    
    data = cell(1, length(resultDirs));
    for idx=1:length(resultDirs)
        resultFiles = dir([resultDirs{idx} '/*.mat']);
        data{idx} = cell(1, length(resultFiles));
        for jdx=1:length(resultFiles)
            name = resultFiles(jdx).name(1:end-4);
            fullName = [resultFiles(jdx).folder '/' name];
            data{idx}{jdx} = load(fullName);
        end
    end

    %% Plot Task Success
    
    figs = [figs, figure];  hold('on');
    plots = zeros(1, length(resultDirs));
    colors = get(gca, 'ColorOrder');
    maxMinRealizationLength = -inf;
    
    for idx=1:length(resultDirs)
        
        nRealizations = length(data{idx});
        
        % determine length of shortest learning curve in this scenario
        minRealizationLength = inf;
        for jdx=1:nRealizations
            if length(data{idx}{jdx}.nPlacedObjects) < minRealizationLength
                minRealizationLength = length(data{idx}{jdx}.nPlacedObjects);
            end
        end
        
        % determine longest plot in comparison
        if minRealizationLength > maxMinRealizationLength
            maxMinRealizationLength = minRealizationLength;
        end
        
        nPlaced = zeros(nRealizations, minRealizationLength);
    
        for jdx=1:nRealizations
            nPlaced(jdx, 1:minRealizationLength) = ...
                data{idx}{jdx}.nPlacedObjects(1:minRealizationLength);
        end

        uP = mean(nPlaced, 1);
        sP = std(nPlaced, 0, 1);

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
    ylabel('Number of Objects Placed', 'FontWeight', 'bold');
    
    if ~isempty(resultLegend)
        legend(plots, resultLegend);
        legend('Location', 'best');
        legend('boxoff');
    end
    
    %% Print Quantitative Results
    
    disp('PLACES -------------------------------------------------------');
    for idx=1:length(resultDirs)
        nRealizations = length(data{idx});
        nEpisodes = 0; totalPlaced = 0;
        for jdx=1:nRealizations
            totalPlaced = totalPlaced + sum(data{idx}{jdx}. ...
                nPlacedObjects(data{idx}{jdx}.unbiasOnEpisode+1:end));
            nEpisodes = nEpisodes + length(data{idx}{jdx}. ...
                nPlacedObjects(data{idx}{jdx}.unbiasOnEpisode+1:end));
        end
        disp([resultLegend{idx} ': ' num2str(totalPlaced/nEpisodes) '.']);
    end
    
    disp('TIME ---------------------------------------------------------');
    for idx=1:length(resultDirs)
        nRealizations = length(data{idx});
        totalTime = 0;
        for jdx=1:nRealizations
            totalTime = totalTime + sum(data{idx}{jdx}.episodeTime);
        end
        disp([resultLegend{idx} ': ' num2str(totalTime/nRealizations/3600) ' hours.']);
    end
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    