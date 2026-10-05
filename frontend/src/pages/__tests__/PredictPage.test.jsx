import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return { ...actual, useNavigate: () => mockNavigate };
});

vi.mock('../../utils/pendingUpload', () => ({
  setPendingUploadFiles: vi.fn(),
}));

import { setPendingUploadFiles } from '../../utils/pendingUpload';
import PredictPage from '../PredictPage';

function makeFile(name) {
  return new File(['dummy content'], name, { type: 'application/octet-stream' });
}

function renderPage() {
  render(
    <MemoryRouter>
      <PredictPage />
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe('PredictPage upload flow', () => {
  it('disables submit until all four required modalities are selected', async () => {
    const user = userEvent.setup();
    renderPage();

    const submitButton = screen.getByRole('button', { name: /run segmentation/i });
    expect(submitButton).toBeDisabled();

    // Required modality inputs are rendered first, in order: t1, t1ce, t2, flair.
    const fileInputs = document.querySelectorAll('input[type="file"]');
    expect(fileInputs.length).toBeGreaterThanOrEqual(4);

    await user.upload(fileInputs[0], makeFile('t1.nii.gz'));
    expect(submitButton).toBeDisabled();

    await user.upload(fileInputs[1], makeFile('t1ce.nii.gz'));
    await user.upload(fileInputs[2], makeFile('t2.nii.gz'));
    expect(submitButton).toBeDisabled();

    await user.upload(fileInputs[3], makeFile('flair.nii.gz'));
    expect(submitButton).not.toBeDisabled();
  });

  it('stores the selected files and navigates to /processing on submit', async () => {
    const user = userEvent.setup();
    renderPage();

    const fileInputs = document.querySelectorAll('input[type="file"]');
    await user.upload(fileInputs[0], makeFile('t1.nii.gz'));
    await user.upload(fileInputs[1], makeFile('t1ce.nii.gz'));
    await user.upload(fileInputs[2], makeFile('t2.nii.gz'));
    await user.upload(fileInputs[3], makeFile('flair.nii.gz'));

    await user.click(screen.getByRole('button', { name: /run segmentation/i }));

    expect(setPendingUploadFiles).toHaveBeenCalledTimes(1);
    const storedFiles = setPendingUploadFiles.mock.calls[0][0];
    expect(storedFiles.t1.name).toBe('t1.nii.gz');
    expect(storedFiles.flair.name).toBe('flair.nii.gz');

    expect(mockNavigate).toHaveBeenCalledWith(
      '/processing',
      expect.objectContaining({ state: expect.objectContaining({ files: storedFiles }) })
    );
  });

  it('flags duplicate uploads across modalities instead of submitting', async () => {
    const user = userEvent.setup();
    renderPage();

    const sameFile = makeFile('duplicate.nii.gz');
    const fileInputs = document.querySelectorAll('input[type="file"]');

    // Same name+size uploaded for two different modalities.
    await user.upload(fileInputs[0], sameFile);
    await user.upload(fileInputs[1], sameFile);
    await user.upload(fileInputs[2], makeFile('t2.nii.gz'));
    await user.upload(fileInputs[3], makeFile('flair.nii.gz'));

    await user.click(screen.getByRole('button', { name: /run segmentation/i }));

    expect(await screen.findByText(/please fix the following errors/i)).toBeInTheDocument();
    expect(screen.getAllByText(/duplicate of/i).length).toBeGreaterThan(0);
    expect(setPendingUploadFiles).not.toHaveBeenCalled();
    expect(mockNavigate).not.toHaveBeenCalled();
  });
});
