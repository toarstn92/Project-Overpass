class NHSUKInlineFeedbackComponent {
    feedbackForm = null;
    isInitialized = false;
    feedbackSection = null;
    feedbackSessionStorageKey = null;

    init() {
        if (this.isInitialized) return;

        const feedbackSection = document.querySelector('#nhsuk-feedback-section');
        const formElement = feedbackSection ? feedbackSection.querySelector('.nhsuk-feedback-form') : null;
        const pageId = formElement ? formElement.dataset.pageId : null;
        const feedbackButtons = formElement ? formElement.querySelectorAll('.nhsuk-feedback-form__button') : [];

        if (!feedbackSection || !formElement || !pageId) {
            return;
        }

        for (const button of feedbackButtons) {
            button.addEventListener('click', this.handleFeedback);
        }

        this.feedbackSection = feedbackSection;
        this.feedbackForm = formElement;
        this.feedbackSessionStorageKey = `nhsuk-feedback-submitted-${pageId}`;

        if (!this.checkCookieConsent()) {
            document.addEventListener('click', event => {
                if (event.target.id === 'nhsuk-cookie-banner__link_accept_analytics') {
                    this.checkCookieAndRender();
                }
            });
            return;
        }

        this.renderFeedbackComponent();
        this.isInitialized = true;
    }

    checkCookieAndRender = () => {
        if (this.checkCookieConsent()) {
            this.renderFeedbackComponent();
        }
    }

    renderFeedbackComponent = () => {
        this.feedbackSection.style.display = 'block';
        if(!sessionStorage.getItem(this.feedbackSessionStorageKey)) {
            this.feedbackForm.style.display = 'block';
            const thankYouElement = document.querySelector('.nhsuk-feedback-form__thank-you');
            if (thankYouElement) {
                thankYouElement.setAttribute('aria-hidden', 'true');
            }
            return;
        }
        this.feedbackForm.style.display = 'none';
        this.showThankYouMessage();
    }

    checkCookieConsent = () => {
        return typeof NHSCookieConsent !== 'undefined' && NHSCookieConsent ? NHSCookieConsent.getStatistics() : false;
    }

    handleFeedback = (event) => {
        event.preventDefault();
        sessionStorage.setItem(this.feedbackSessionStorageKey, 'true');
        this.feedbackForm.style.display = 'none';
        this.showThankYouMessage(true);
    }

    showThankYouMessage = (shouldSetFocus = false) => {
        const thankYouElement = document.querySelector('.nhsuk-feedback-form__thank-you');
        if (thankYouElement) {
            thankYouElement.style.display = 'block';
            thankYouElement.removeAttribute('aria-hidden');
            
            const heading = thankYouElement.querySelector('.nhsuk-card__heading');
            if (heading && shouldSetFocus) {
                heading.setAttribute('tabindex', '-1');
                heading.focus();
            }
        }
    }
}

const feedbackComponent = new NHSUKInlineFeedbackComponent();
feedbackComponent.init();
