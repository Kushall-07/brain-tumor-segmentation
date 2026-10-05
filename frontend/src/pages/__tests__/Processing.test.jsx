import { act, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => ({ state: null }),
  };
});

vi.mock('../../services/predictionService', () => ({
  default: {
    startPrediction: vi.fn(),
    getPredictionStatus: vi.fn(),
  },
}));

vi.mock('../../utils/pendingUpload', () => ({
  consumePendingUploadFiles: vi.fn(),
}));

import predictionService from '../../services/predictionService';
import { consumePendingUploadFiles } from '../../utils/pendingUpload';
import Processing from '../Processing';

const fakeFiles = {
  t1: new File(['a'], 't1.nii.gz'),
  t1ce: new File(['a'], 't1ce.nii.gz'),
  t2: new File(['a'], 't2.nii.gz'),
  flair: new File(['a'], 'flair.nii.gz'),
  seg: null,
};

function renderPage() {
  render(
    <MemoryRouter>
      <Processing />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  consumePendingUploadFiles.mockReturnValue(fakeFiles);
});

afterEach(() => {
  vi.useRealTimers();
});

describe('Processing job polling', () => {
  it('polls for status until the job completes, then navigates to /results', async () => {
    predictionService.startPrediction.mockResolvedValue({ job_id: 'job-1', status: 'processing' });
    predictionService.getPredictionStatus
      .mockResolvedValueOnce({ status: 'processing', stage: 'validation', message: 'Validating MRI volumes' })
      .mockResolvedValueOnce({ status: 'completed', result: { mask_path: 'outputs/predictions/x/mask.nii.gz' } });

    renderPage();

    // First poll resolves immediately after the job is created.
    await screen.findByText('Validating MRI volumes');
    expect(predictionService.getPredictionStatus).toHaveBeenCalledTimes(1);

    // Second poll is scheduled ~1s later (POLL_INITIAL_INTERVAL_MS) — real
    // timers here keep the recursive setTimeout chain in Processing.jsx
    // genuinely exercised rather than mocked away.
    await waitFor(
      () => expect(predictionService.getPredictionStatus).toHaveBeenCalledTimes(2),
      { timeout: 3000 }
    );

    await waitFor(() =>
      expect(mockNavigate).toHaveBeenCalledWith(
        '/results',
        expect.objectContaining({
          state: expect.objectContaining({
            result: { mask_path: 'outputs/predictions/x/mask.nii.gz' },
          }),
        })
      )
    );
  });

  it('shows an error state when the job fails', async () => {
    predictionService.startPrediction.mockResolvedValue({ job_id: 'job-1', status: 'processing' });
    predictionService.getPredictionStatus.mockResolvedValue({
      status: 'failed',
      error: 'Inference execution failed',
    });

    renderPage();

    expect(await screen.findByText(/could not be completed/i)).toBeInTheDocument();
    expect(screen.getByText('Inference execution failed')).toBeInTheDocument();
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('shows a timeout error if the job never finishes', async () => {
    // The component's timeout check compares Date.now() against when
    // polling started, so Date must be faked alongside the timers —
    // otherwise real wall-clock time (a few ms of test execution) never
    // reaches POLL_TIMEOUT_MS no matter how far timers are advanced.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });

    predictionService.startPrediction.mockResolvedValue({ job_id: 'job-1', status: 'processing' });
    predictionService.getPredictionStatus.mockResolvedValue({
      status: 'processing',
      stage: 'model_inference',
      message: 'Running 3D U-Net segmentation',
    });

    renderPage();

    // Drain the initial startPrediction -> beginPolling -> first pollJobStatus
    // microtask chain (pure promise resolutions, no timers yet) before
    // advancing virtual time, so the first setTimeout actually gets
    // registered on the fake clock. Wrapped in act() so React commits the
    // state updates these resolutions trigger.
    await act(async () => {
      for (let i = 0; i < 20; i++) {
        await Promise.resolve();
      }
    });

    // Past POLL_TIMEOUT_MS (10 minutes) with no completion/failure. Also
    // wrapped in act() — otherwise the state updates from each recursive
    // poll tick apply without React committing/flushing them for assertions.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(11 * 60 * 1000);
    });

    expect(screen.getByText(/could not be completed/i)).toBeInTheDocument();
    expect(screen.getByText(/timed out/i)).toBeInTheDocument();
    expect(mockNavigate).not.toHaveBeenCalled();
  });
});
