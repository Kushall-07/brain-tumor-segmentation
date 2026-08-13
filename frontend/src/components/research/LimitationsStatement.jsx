export default function LimitationsStatement() {
  return (
    <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
      <h2 className="font-serif text-xl font-semibold text-ink mb-3 pb-3 border-b border-sepia-border">
        Limitations
      </h2>
      <p className="text-sm text-ink-body leading-relaxed">
        The model was developed and evaluated using the BraTS brain tumor dataset and therefore its
        performance may not generalize to other datasets, scanners, imaging protocols, tumor types, or
        clinical populations. The system is intended for research and educational purposes and requires
        further external validation before any clinical application.
      </p>
    </div>
  );
}
