function PlotResultsMultiple()

    %% Parameters
    
    resultFiles = dir('../results-*.mat');
    episodeBlock = 1000;
    
    saveFilePrefix = '2019-02-01-B';
    saveFilePostfixes = {'AverageReturn', 'GraspReturn', 'PlaceReturn', ...
        'ReturnStatistics', 'Loss'};
    figsToSave = [];

    %% Load

    close('all');
    if length(resultFiles) < 2
        disp(['Only ' num2str(length(resultFiles)) ' result files.']);
        return;
    end
    
    data = cell(1, length(resultFiles));
    resultFileNames = cell(1, length(resultFiles));
    for idx=1:length(resultFiles)
        name = resultFiles(idx).name(1:end-4);
        fullName = [resultFiles(idx).folder '/' name];
        resultFileNames{idx} = name;
        data{idx} = load(fullName);
    end
    figs = [];

    %% Plot Metrics
    
    figs = [figs PlotMultiple(resultFileNames, data, 'episodeReturn', ...
        episodeBlock, 0, 6, 'Return')];
    
    %% Time
    
    disp('Time ---------------------------------------------------------');
    for idx=1:length(resultFileNames)
        disp([resultFileNames{idx} ': ' ...
            num2str(sum(data{idx}.episodeTime) / 3600) 'h.']);
    end
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    
end
    
function fig = PlotMultiple(fileNames, data, metricName, episodeBlock, ...
    worstReturn, bestReturn, titleName)

    fig = figure;  hold('on');
    nRealizations = length(fileNames);
    maxEpisodes = -Inf;
    
    for idx=1:nRealizations
        metric = eval(['data{idx}.', metricName]);
        if length(metric) > maxEpisodes, maxEpisodes = length(metric); end
        rVis = reshape(metric, [episodeBlock, length(metric)/episodeBlock]);
        rVis = mean(rVis, 1);
        episode = (0:episodeBlock:length(metric)-1);
        plot(episode, rVis, '-', 'linewidth', 2);
    end
    
    xlim([0, maxEpisodes-episodeBlock]); ylim([worstReturn, bestReturn]);
    xlabel('Episode'); ylabel(titleName); grid('on');
    title([titleName ' Averaged over ' num2str(episodeBlock) ...
        '-Episode Blocks']);
    
    if nRealizations <= 10
        legend(fileNames);
        legend('Location', 'best');
        legend('boxoff');
    end
end