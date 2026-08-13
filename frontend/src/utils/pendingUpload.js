/** In-memory store for upload files between Predict and Processing routes. */

let pendingUploadFiles = null;

export function setPendingUploadFiles(files) {
  pendingUploadFiles = files;
}

export function consumePendingUploadFiles() {
  const files = pendingUploadFiles;
  pendingUploadFiles = null;
  return files;
}
