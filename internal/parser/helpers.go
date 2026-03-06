package parser

func qualifiedMemberName(receiver, name string) string {
	if receiver == "" {
		return name
	}
	return receiver + "." + name
}
