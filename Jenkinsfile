@Library('mlops-lib') _

def MODELS = [
    [dataset: 'disease', model: 'logistic_regression'],
    [dataset: 'disease', model: 'random_forest'],
    [dataset: 'boston',  model: 'random_forest']
]

pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
        NOTIFY_TO = 'mohsinsaeed092005@gmail.com'
    }

    stages {
        stage('Setup') {
            steps { setupEnv() }
        }

        // ---------------- DEV (push to dev) ----------------
        stage('Dev Pipeline') {
            when { branch 'dev' }
            stages {
                stage('Data Ingest') {
                    steps { ingestData() }
                }
                stage('Model Train + Register (Challenger)') {
                    steps {
                        script {
                            for (m in MODELS) { trainModel(m.dataset, m.model, 'Challenger') }
                        }
                    }
                }
                stage('Model Deploy (dev)') {
                    steps {
                        script {
                            for (m in MODELS) { deployModel(m.dataset, m.model, 'Challenger', 'dev') }
                        }
                    }
                }
                stage('Model Test') {
                    steps {
                        script {
                            for (m in MODELS) { testModel(m.dataset, m.model, 'Challenger') }
                        }
                    }
                }
            }
        }

        // ---------------- PRE-PROD (merge to main) ----------------
        stage('Pre-prod Pipeline') {
            when { branch 'main' }
            stages {
                stage('Load Challenger -> alias Challenger-pre-test') {
                    steps {
                        script {
                            for (m in MODELS) {
                                promoteModel("${m.dataset}_${m.model}", 'Challenger', 'Challenger-pre-test')
                            }
                        }
                    }
                }
                stage('Model Deploy (preprod)') {
                    steps {
                        script {
                            for (m in MODELS) { deployModel(m.dataset, m.model, 'Challenger-pre-test', 'preprod') }
                        }
                    }
                }
                stage('Model Test') {
                    steps {
                        script {
                            for (m in MODELS) { testModel(m.dataset, m.model, 'Challenger-pre-test') }
                        }
                    }
                }
                stage('Update alias -> Challenger-post-test') {
                    steps {
                        script {
                            for (m in MODELS) {
                                promoteModel("${m.dataset}_${m.model}", 'Challenger-pre-test', 'Challenger-post-test')
                            }
                        }
                    }
                }
            }
        }

        // ---------------- PROD (release tag) ----------------
        stage('Prod Pipeline') {
            when { buildingTag() }
            stages {
                stage('Load Challenger-post-test -> alias Champion') {
                    steps {
                        script {
                            for (m in MODELS) {
                                promoteModel("${m.dataset}_${m.model}", 'Challenger-post-test', 'Champion')
                            }
                        }
                    }
                }
                stage('Model Deploy (prod)') {
                    steps {
                        script {
                            for (m in MODELS) { deployModel(m.dataset, m.model, 'Champion', 'prod') }
                        }
                    }
                }
            }
        }
    }

    post {
        success { notifyEmail('SUCCESS', env.NOTIFY_TO) }
        failure { notifyEmail('FAILED', env.NOTIFY_TO) }
    }
}
