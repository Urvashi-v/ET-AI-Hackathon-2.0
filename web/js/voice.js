/**
 * Voice input, via the browser's own Web Speech API.
 *
 * The whole point of this module is the first thing it does: check whether
 * speech recognition actually exists in this browser, and report honestly when
 * it does not. A microphone button that opens a dialog and then quietly does
 * nothing is worse than no button — a technician in a noisy plant will try it,
 * get nothing, and stop trusting the interface.
 *
 * `SpeechRecognition` is genuinely supported in Chrome, Edge and Safari, and
 * genuinely absent in Firefox. In Chrome it also routes audio to a Google
 * server, which matters for a plant with a policy about that, so
 * :func:`describe` says so rather than leaving it to be discovered.
 *
 * There is no fallback implementation. No recording-and-uploading to a
 * transcription service that is not configured, no "voice coming soon" that
 * fills the field with placeholder text. Available or reported unavailable.
 */

const Recognition =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;

/** Is dictation genuinely available in this browser, right now? */
export function isSupported() {
  return Recognition !== null;
}

/**
 * What to tell the user about voice input here.
 *
 * Includes the privacy note because Chrome's implementation sends audio off the
 * device, and a plant that has not approved that needs to know before someone
 * dictates an incident description into it.
 */
export function describe() {
  if (!Recognition) {
    return {
      available: false,
      reason:
        'This browser has no speech recognition API. Firefox does not implement it; ' +
        'Chrome, Edge and Safari do. Nothing is simulated in its place.',
      privacyNote: null,
    };
  }
  const isChromium = 'webkitSpeechRecognition' in window && !('SpeechRecognition' in window);
  return {
    available: true,
    reason: null,
    privacyNote: isChromium
      ? 'This browser sends audio to a Google speech service for recognition. ' +
        'Nothing is recorded or stored by this application.'
      : 'Recognition is handled by the browser. Nothing is recorded or stored by ' +
        'this application.',
  };
}

/**
 * One dictation attempt.
 *
 * @param {{lang?: string, onPartial?: (text: string) => void}} options
 * @returns {{promise: Promise<string>, stop: () => void}}
 *
 * Returns a stop handle as well as the promise, so the UI can offer a cancel
 * button — a recogniser left listening in a loud plant will keep returning
 * noise until something stops it.
 */
export function listen({ lang = 'en-IN', onPartial } = {}) {
  if (!Recognition) {
    return {
      promise: Promise.reject(new Error('Speech recognition is not available in this browser.')),
      stop: () => {},
    };
  }

  const recognition = new Recognition();
  recognition.lang = lang;
  // Interim results give the user visible feedback that the microphone is live,
  // which in a noisy environment is the difference between waiting and giving up.
  recognition.interimResults = Boolean(onPartial);
  recognition.continuous = false;
  recognition.maxAlternatives = 1;

  let settled = false;
  const promise = new Promise((resolve, reject) => {
    recognition.addEventListener('result', (event) => {
      let finalText = '';
      let partial = '';
      for (const result of event.results) {
        if (result.isFinal) finalText += result[0].transcript;
        else partial += result[0].transcript;
      }
      if (partial && onPartial) onPartial(partial);
      if (finalText) {
        settled = true;
        resolve(finalText.trim());
      }
    });

    recognition.addEventListener('error', (event) => {
      settled = true;
      // Mapped to what the user can do about it. "not-allowed" means they
      // declined the microphone prompt, which is a choice, not a fault.
      const messages = {
        'not-allowed': 'Microphone access was denied. Allow it in the browser to dictate.',
        'service-not-allowed': 'The browser blocked its speech service.',
        'no-speech': 'Nothing was heard. Try again closer to the microphone.',
        network: 'Speech recognition needs a network connection and there is none.',
        aborted: 'Dictation was cancelled.',
      };
      reject(new Error(messages[event.error] || `Speech recognition failed: ${event.error}`));
    });

    recognition.addEventListener('end', () => {
      // Ended without a final result: silence, or a stop before anything was
      // said. Resolving empty rather than rejecting keeps "said nothing" from
      // being presented to the user as an error.
      if (!settled) resolve('');
    });
  });

  try {
    recognition.start();
  } catch (error) {
    return { promise: Promise.reject(error), stop: () => {} };
  }

  return { promise, stop: () => recognition.abort() };
}
