function PlotResultsSingle()

    %% Parameters
    
    resultsFileName = '../results-0.mat';
    episodeBlock = 10000;
    
    saveFilePrefix = '2019-02-07';
    saveFilePostfixes = {'AverageReturn', 'EpisodeTime'};
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
    
    worstReturn = 0; bestReturn = max(initQ);
    rVis = reshape(nPlacedObjects, [episodeBlock, length(nPlacedObjects)/episodeBlock]);
    rVis = mean(rVis, 1);
    episode = (episodeBlock:episodeBlock:length(nPlacedObjects));
    
    figs = [figs, figure]; hold('on')
    plot(episode, rVis, '-', 'linewidth', 2);
    
    xlim([episode(1), episode(end)]); ylim([worstReturn, bestReturn]); grid('on');
    xlabel('Episode'); ylabel('Average Return');
    title(['Return Averaged over 1 Realization and ' num2str(episodeBlock) '-Episode Blocks']);
    
    %% Plot Run Time
    
    figs = [figs, figure]; hold('on');
    plot(episodeTime, '.', 'linewidth', 2);
    plot(ones(1,length(episodeTime))*mean(episodeTime), '-', 'linewidth', 2);
    xlabel('Episode'); ylabel('Time (s)'); grid('on'); title('Episode Time');
    
    disp(['Mean episode time: ' num2str(mean(episodeTime)) ' seconds.']);
    disp(['Total time: ' num2str(sum(episodeTime)/3600) ' hours.']);
    
    %% Saving Images
    
    for idx=1:length(figsToSave)
        saveas(figs(figsToSave(idx)), ['../../Notebook/figures-4/' ...
            saveFilePrefix '-' saveFilePostfixes{figsToSave(idx)} '.png']);
    end
    