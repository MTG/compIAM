var actions = [
  {
    class: 'fas.fa-minus',
    title: 'Reduce annotation end by 0.010s',
    action: (annotation, i, annotations, opts) => {
      var next;
      var delta = 0.010;
      annotation.end -= delta;

      if (opts.linkEndpoints) {
        next = annotations[i + 1];
        next && (next.start -= delta);
      }
    }
  },
  {
    class: 'fas.fa-plus',
    title: 'Increase annotation end by 0.010s',
    action: (annotation, i, annotations, opts) => {
      var next;
      var delta = 0.010;
      annotation.end += delta;

      if (opts.linkEndpoints) {
        next = annotations[i + 1];
        next && (next.start += delta);
      }
    }
  },
  {
    class: 'fas.fa-cut',
    title: 'Split annotation in half',
    action: (annotation, i, annotations) => {
      const halfDuration = (annotation.end - annotation.start) / 2;

      annotations.splice(i + 1, 0, {
        id: 'test',
        start: annotation.end - halfDuration,
        end: annotation.end,
        lines: ['----'],
        lang: 'en',
      });

      annotation.end = annotation.start + halfDuration;
    }
  },
  {
    class: 'fas.fa-trash',
    title: 'Delete annotation',
    action: (annotation, i, annotations) => {
      annotations.splice(i, 1);
    }
  }
];
