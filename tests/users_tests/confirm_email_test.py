from fastapi.testclient import TestClient
from unittest.mock import patch, ANY
from app import app
import pytest

client = TestClient(app)

def test_confirm_user_signup_success():
    with patch('services.auth_service.get_secret_hash', return_value='fake-secret-hash') as mock_hash, \
         patch('services.auth_service.ses_client') as mock_ses, \
         patch('services.auth_service.incognito_client') as mock_cognito:

        mock_ses.verify_email_identity.return_value = None
        mock_cognito.confirm_sign_up.return_value = {'status': 'SUCCESS'}

        from services.auth_service import confirm_user_signup
        result = confirm_user_signup('test@example.com', '123456')

        mock_hash.assert_called_once_with('test@example.com', ANY, ANY)
        mock_ses.verify_email_identity.assert_called_once_with(EmailAddress='test@example.com')
        mock_cognito.confirm_sign_up.assert_called_once_with(
            ClientId=ANY,
            Username='test@example.com',
            ConfirmationCode='123456',
            SecretHash='fake-secret-hash'
        )
        assert result == {'status': 'SUCCESS'}

def test_confirm_user_signup_cognito_error():
    with patch('services.auth_service.get_secret_hash', return_value='fake-secret-hash'), \
         patch('services.auth_service.ses_client') as mock_ses, \
         patch('services.auth_service.incognito_client') as mock_cognito:

        mock_ses.verify_email_identity.return_value = None
        mock_cognito.confirm_sign_up.side_effect = Exception("Cognito error")

        from services.auth_service import confirm_user_signup
        with pytest.raises(Exception) as excinfo:
            confirm_user_signup('test@example.com', '123456')
        assert "Cognito error" in str(excinfo.value)


def test_confirm_user_signup_missing_confirmation_code():
    with patch('services.auth_service.get_secret_hash', return_value='fake-secret-hash'), \
         patch('services.auth_service.ses_client') as mock_ses, \
         patch('services.auth_service.incognito_client') as mock_cognito:

        mock_ses.verify_email_identity.return_value = None
        # Simulate Cognito raising an error for missing confirmation code
        mock_cognito.confirm_sign_up.side_effect = Exception("Missing confirmation code")

        from services.auth_service import confirm_user_signup
        with pytest.raises(Exception) as excinfo:
            confirm_user_signup('test@example.com', None)  # or use ""
        assert "Missing confirmation code" in str(excinfo.value)