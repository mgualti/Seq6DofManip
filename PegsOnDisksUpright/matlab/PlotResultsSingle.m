function PlotResultsSingle()

    %% Parameters
    
    resultsFileName = '../results-0.mat';
    episodeBlock = 1000; worstReturn = 0; bestReturn = 4;
    
    saveFilePrefix = '2019-01-04';
    saveFilePostfixes = {'AverageReturn', 'Exploration', 'DatabaseSize', 'EpisodeTime', 'Loss'};
    figsToSave = [];

    %% Load

    close('all');

    if ~exist(resultsFileName, 'file')
        disp([resultsFileName ' not found.']);
        return
    end
    load(resultsFileName);
    figs = [];

    %% Plot Return
    
    rVis = reshape(episodeReturn, [episodeBlock, length(episodeReturn)/episodeBlock]);
    rVis = mean(rVis, 1);
    episode = (0:episodeBlock:length(episodeReturn)-1);
    
    figs = [figs, figure]; hold('on')
    plot(episode, rVis, '-x', 'linewidth', 2);
    
    ylim([worstReturn, bestReturn]); grid('on');
    xlabel('Episode'); ylabel('Average Return');
    title(['Return Averaged over 1 Realization and ' num2str(episodeBlock) '-Episode Blocks']);
    
    unbiasedReturn = mean(episodeReturn(unbiasOnEpisode+1:end));
    disp(['Average unbiased return: ' num2str(unbiasedReturn) '.']);
    
    %% Plot Exploration and Experience Database Size
    
    figs = [figs, figure]; hold('on');
    episode = (0:tMax*length(episodeReturn)-1) / tMax;
    plot(episode, timeStepEpsilon, '.', 'linewidth', 2);
    xlabel('Episode'); ylabel('\epsilon'); grid('on'); title('Exploration');
    
    figs = [figs, figure]; hold('on');
    plot(databaseSize, '-', 'linewidth', 2);
    xlabel('Episode'); ylabel('Database Size'); grid('on'); title('Database Size');
    
    %% Plot Run Time
    
    figs = [figs, figure]; hold('on');
    plot(episodeTime, 'x', 'linewidth', 2);
    plot(ones(1,length(episodeTime))*mean(episodeTime), '-', 'linewidth', 2);
    xlabel('Episode'); ylabel('Time (s)'); grid('on'); title('Episode Time');
    
    disp(['Mean episode time: ' num2str(mean(episodeTime)) ' seconds.']);
    disp(['Total time: ' num2str(sum(episodeTime)/3600) ' hours.']);
    
    
    %% Plot Loss
    
    figs = [figs, figure]; hold('on');
    episode = (0:trainEvery:length(episodeReturn)-1);
    for level=1:size(losses, 2)
        loss = losses(:,level);
        plot(episode, loss, '-x', 'linewidth', 2);
    end
    xlabel('Episode'); ylabel('Loss'); grid('on');
    title('Average Loss Per Training Round');
    legend('Level-1','Level-2','Level-3');
    legend('Location', 'best'); legend('boxoff');
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    